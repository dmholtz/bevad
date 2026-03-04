let get_wasm_bindgen = null;
let _wasm_module = null;
async function fetch_viewer_js(base_url) {
    // @ts-ignore
    return (await import("./re_viewer.js")).default;
}
async function fetch_viewer_wasm(base_url) {
    //!<INLINE-MARKER-OPEN>
    if (base_url) {
        return fetch(new URL("./re_viewer_bg.wasm", base_url));
    }
    else {
        return fetch(new URL("./re_viewer_bg.wasm", import.meta.url));
    }
    //!<INLINE-MARKER-CLOSE>
}
async function load(base_url) {
    // instantiate wbg globals+module for every invocation of `load`,
    // but don't load the JS/Wasm source every time
    if (!get_wasm_bindgen || !_wasm_module) {
        [get_wasm_bindgen, _wasm_module] = await Promise.all([
            fetch_viewer_js(base_url),
            WebAssembly.compileStreaming(fetch_viewer_wasm(base_url)),
        ]);
    }
    let bindgen = get_wasm_bindgen();
    await bindgen({ module_or_path: _wasm_module });
    return class extends bindgen.WebHandle {
        free() {
            super.free();
            // @ts-ignore
            bindgen.deinit();
        }
    };
}
let _minimize_current_fullscreen_viewer = null;
function randomId() {
    const bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);
    return Array.from(bytes)
        .map((byte) => byte.toString(16).padStart(2, "0"))
        .join("");
}
function delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}
/**
 * Rerun Web Viewer
 *
 * ```ts
 * const viewer = new WebViewer();
 * await viewer.start();
 * ```
 *
 * Data may be provided to the Viewer as:
 * - An HTTP file URL, e.g. `viewer.start("https://app.rerun.io/version/0.30.0/examples/dna.rrd")`
 * - A Rerun gRPC URL, e.g. `viewer.start("rerun+http://127.0.0.1:9876/proxy")`
 * - A stream of log messages, via {@link WebViewer.open_channel}.
 *
 * Callbacks may be attached for various events using {@link WebViewer.on}:
 *
 * ```ts
 * viewer.on("time_update", (time) => console.log(`current time: {time}`));
 * ```
 *
 * For the full list of available events, see {@link ViewerEvent}.
 */
export class WebViewer {
    #id = randomId();
    // NOTE: Using the handle requires wrapping all calls to its methods in try/catch.
    //       On failure, call `this.stop` to prevent a memory leak, then re-throw the error.
    #handle = null;
    #canvas = null;
    #state = "stopped";
    #fullscreen = false;
    #allow_fullscreen = false;
    constructor() {
        injectStyle();
        setupGlobalEventListeners();
    }
    /**
     * Start the viewer.
     *
     * @param rrd URLs to `.rrd` files or gRPC connections to our SDK.
     * @param parent The element to attach the canvas onto.
     * @param options Web Viewer configuration.
     */
    async start(rrd, parent, options) {
        parent ??= document.body;
        options ??= {};
        options = options ? { ...options } : options;
        this.#allow_fullscreen = options.allow_fullscreen || false;
        if (this.#state !== "stopped")
            return;
        this.#state = "starting";
        this.#canvas = document.createElement("canvas");
        this.#canvas.style.width = options.width ?? "640px";
        this.#canvas.style.height = options.height ?? "360px";
        parent.append(this.#canvas);
        // Show loading spinner
        const loader = document.createElement("div");
        loader.id = "rerun-loader";
        loader.innerHTML = `
      <style>
        @keyframes rerun-spin { to { transform: rotate(360deg); } }
      </style>
      <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; background-color: #1c1c1c; font-family: sans-serif; color: white;">
        <div style="width: 40px; height: 40px; border: 3px solid #444; border-top-color: white; border-radius: 50%; animation: rerun-spin 1s linear infinite;"></div>
        <div style="margin-top: 16px;">Loading Rerun…</div>
      </div>
    `;
        loader.style.position = "absolute";
        loader.style.inset = "0";
        parent.style.position = "relative";
        parent.append(loader);
        // This yield appears to be necessary to ensure that the canvas is attached to the DOM
        // and visible. Without it we get occasionally get a panic about a failure to find a canvas
        // element with the given ID.
        await delay(0);
        let base_url = options?.base_url;
        if (base_url) {
            delete options.base_url;
        }
        let WebHandle_class;
        try {
            WebHandle_class = await load(base_url);
        }
        catch (e) {
            loader.remove();
            this.#fail("Failed to load rerun", String(e));
            throw e;
        }
        if (this.#state !== "starting")
            return;
        const fullscreen = this.#allow_fullscreen
            ? {
                get_state: () => this.#fullscreen,
                on_toggle: () => this.toggle_fullscreen(),
            }
            : undefined;
        const on_viewer_event = (event_json) => {
            // for notebooks/gradio, we can avoid a whole layer
            // of serde by sending over the raw json directly,
            // which will be deserialized in Python instead
            this.#dispatch_raw_event(event_json);
            // for JS users, we dispatch the parsed event
            let event = JSON.parse(event_json);
            this.#dispatch_event(event.type, event);
        };
        this.#handle = new WebHandle_class({
            ...options,
            fullscreen,
            on_viewer_event,
        });
        try {
            await this.#handle.start(this.#canvas);
        }
        catch (e) {
            loader.remove();
            this.#fail("Failed to start", String(e));
            throw e;
        }
        if (this.#state !== "starting")
            return;
        loader.remove();
        this.#state = "ready";
        this.#dispatch_event("ready");
        if (rrd) {
            this.open(rrd);
        }
        let self = this;
        function check_for_panic() {
            if (self.#handle?.has_panicked()) {
                self.#fail("Rerun has crashed.", self.#handle?.panic_message());
            }
            else {
                let delay_ms = 1000;
                setTimeout(check_for_panic, delay_ms);
            }
        }
        check_for_panic();
        return;
    }
    #raw_events = new Set();
    #dispatch_raw_event(event_json) {
        for (const callback of this.#raw_events) {
            callback(event_json);
        }
    }
    /** Internal interface */
    // NOTE: Callbacks passed to this function must NOT invoke any viewer methods!
    //       The `setTimeout` is omitted to avoid the 1-tick delay, as it is unnecessary,
    //       because this is only meant to be used for sending events to Jupyter/Gradio.
    //
    // Do not change this without searching for grepping for usage!
    _on_raw_event(callback) {
        this.#raw_events.add(callback);
        return () => this.#raw_events.delete(callback);
    }
    #event_map = new Map();
    #dispatch_event(event, ...args) {
        // Dispatch events on next tick.
        // This is necessary because we may have been called somewhere deep within the viewer's call stack,
        // which means that `app` may be locked. The event will not actually be dispatched until the
        // full call stack has returned or the current task has yielded to the event loop. It does not
        // guarantee that we will be able to acquire the lock here, but it makes it a lot more likely.
        setTimeout(() => {
            const callbacks = this.#event_map.get(event);
            if (callbacks) {
                for (const [callback, { once }] of [...callbacks.entries()]) {
                    callback(...args);
                    if (once)
                        callbacks.delete(callback);
                }
            }
        }, 0);
    }
    on(event, callback) {
        const callbacks = this.#event_map.get(event) ?? new Map();
        callbacks.set(callback, { once: false });
        this.#event_map.set(event, callbacks);
        return () => callbacks.delete(callback);
    }
    once(event, callback) {
        const callbacks = this.#event_map.get(event) ?? new Map();
        callbacks.set(callback, { once: true });
        this.#event_map.set(event, callbacks);
        return () => callbacks.delete(callback);
    }
    off(event, callback) {
        const callbacks = this.#event_map.get(event);
        if (callbacks) {
            callbacks.delete(callback);
        }
        else {
            console.warn("Attempted to call `WebViewer.off` with an unregistered callback. Are you passing in the same function instance?");
        }
    }
    /**
     * The underlying canvas element.
     */
    get canvas() {
        return this.#canvas;
    }
    /**
     * Returns `true` if the viewer is ready to connect to data sources.
     */
    get ready() {
        return this.#state === "ready";
    }
    /**
     * Open a recording.
     *
     * The viewer must have been started via {@link WebViewer.start}.
     *
     * @param rrd URLs to `.rrd` files or gRPC connections to our SDK.
     * @param options
     *        - follow_if_http: Whether Rerun should open the resource in "Following" mode when streaming
     *        from an HTTP url. Defaults to `false`. Ignored for non-HTTP URLs.
     */
    open(rrd, options = {}) {
        if (!this.#handle) {
            throw new Error(`attempted to open \`${rrd}\` in a stopped viewer`);
        }
        const urls = Array.isArray(rrd) ? rrd : [rrd];
        for (const url of urls) {
            try {
                this.#handle.add_receiver(url, options.follow_if_http);
            }
            catch (e) {
                this.#fail("Failed to open recording", String(e));
                throw e;
            }
        }
    }
    /**
     * Close a recording.
     *
     * The viewer must have been started via {@link WebViewer.start}.
     *
     * @param rrd URLs to `.rrd` files or gRPC connections to our SDK.
     */
    close(rrd) {
        if (!this.#handle) {
            throw new Error(`attempted to close \`${rrd}\` in a stopped viewer`);
        }
        const urls = Array.isArray(rrd) ? rrd : [rrd];
        for (const url of urls) {
            try {
                this.#handle.remove_receiver(url);
            }
            catch (e) {
                this.#fail("Failed to close recording", String(e));
                throw e;
            }
        }
    }
    /**
     * Stop the viewer, freeing all associated memory.
     *
     * The same viewer instance may be started multiple times.
     */
    stop() {
        if (this.#state === "stopped")
            return;
        if (this.#allow_fullscreen && this.#canvas && this.#fullscreen) {
            this.#minimize();
        }
        this.#state = "stopped";
        this.#canvas?.remove();
        try {
            this.#handle?.destroy();
            this.#handle?.free();
        }
        catch (e) {
            this.#handle = null;
            throw e;
        }
        this.#canvas = null;
        this.#handle = null;
        this.#fullscreen = false;
        this.#allow_fullscreen = false;
    }
    #fail(message, error_message) {
        console.error("WebViewer failure:", message, error_message);
        if (this.canvas?.parentElement) {
            const parent = this.canvas.parentElement;
            parent.innerHTML = `
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; color: white; font-family: sans-serif; background-color: #1c1c1c;">
          <h1 id="fail-message"></h1>
          <pre id="fail-error" style="text-align: left;"></pre>
          <button id="fail-clear-cache">Clear caches and reload</button>
        </div>
      `;
            document.getElementById("fail-message").textContent = message;
            const errorEl = document.getElementById("fail-error");
            if (error_message) {
                errorEl.textContent = error_message;
            }
            else {
                errorEl.remove();
            }
            document.getElementById("fail-clear-cache").addEventListener("click", async () => {
                if ("caches" in window) {
                    const keys = await caches.keys();
                    await Promise.all(keys.map((key) => caches.delete(key)));
                }
                window.location.reload();
            });
        }
        this.stop();
    }
    /**
     * Opens a new channel for sending log messages.
     *
     * The channel can be used to incrementally push `rrd` chunks into the viewer.
     *
     * @param channel_name used to identify the channel.
     */
    open_channel(channel_name = "rerun-io/web-viewer") {
        if (!this.#handle) {
            throw new Error(`attempted to open channel \"${channel_name}\" in a stopped web viewer`);
        }
        const id = crypto.randomUUID();
        try {
            this.#handle.open_channel(id, channel_name);
        }
        catch (e) {
            this.#fail("Failed to open channel", String(e));
            throw e;
        }
        const on_send = (/** @type {Uint8Array} */ data) => {
            if (!this.#handle) {
                throw new Error(`attempted to send data through channel \"${channel_name}\" to a stopped web viewer`);
            }
            try {
                this.#handle.send_rrd_to_channel(id, data);
            }
            catch (e) {
                this.#fail("Failed to send data", String(e));
                throw e;
            }
        };
        const on_send_table = (/** @type {Uint8Array} */ data) => {
            if (!this.#handle) {
                throw new Error(`attempted to send data through channel \"${channel_name}\" to a stopped web viewer`);
            }
            try {
                this.#handle.send_table_to_channel(id, data);
            }
            catch (e) {
                this.#fail("Failed to send table", String(e));
                throw e;
            }
        };
        const on_close = () => {
            if (!this.#handle) {
                throw new Error(`attempted to send data through channel \"${channel_name}\" to a stopped web viewer`);
            }
            try {
                this.#handle.close_channel(id);
            }
            catch (e) {
                this.#fail("Failed to close channel", String(e));
                throw e;
            }
        };
        const get_state = () => this.#state;
        return new LogChannel(on_send, on_send_table, on_close, get_state);
    }
    /**
     * Force a panel to a specific state.
     *
     * @param panel which panel to configure
     * @param state which state to force the panel into
     */
    override_panel_state(panel, state) {
        if (!this.#handle) {
            throw new Error(`attempted to set ${panel} panel to ${state} in a stopped web viewer`);
        }
        try {
            this.#handle.override_panel_state(panel, state);
        }
        catch (e) {
            this.#fail("Failed to override panel state", String(e));
            throw e;
        }
    }
    /**
     * Toggle panel overrides set via `override_panel_state`.
     *
     * @param value set to a specific value. Toggles the previous value if not provided.
     */
    toggle_panel_overrides(value) {
        if (!this.#handle) {
            throw new Error(`attempted to toggle panel overrides in a stopped web viewer`);
        }
        try {
            this.#handle.toggle_panel_overrides(value);
        }
        catch (e) {
            this.#fail("Failed to toggle panel overrides", String(e));
            throw e;
        }
    }
    /**
     * Get the active recording id.
     */
    get_active_recording_id() {
        if (!this.#handle) {
            throw new Error(`attempted to get active recording id in a stopped web viewer`);
        }
        return this.#handle.get_active_recording_id() ?? null;
    }
    /**
     * Set the active recording id.
     *
     * This is the same as clicking on the recording in the Viewer's left panel.
     */
    set_active_recording_id(value) {
        if (!this.#handle) {
            throw new Error(`attempted to set active recording id to ${value} in a stopped web viewer`);
        }
        this.#handle.set_active_recording_id(value);
    }
    /**
     * Get the play state.
     *
     * This always returns `false` if the recording can't be found.
     */
    get_playing(recording_id) {
        if (!this.#handle) {
            throw new Error(`attempted to get play state in a stopped web viewer`);
        }
        return this.#handle.get_playing(recording_id) || false;
    }
    /**
     * Set the play state.
     *
     * This does nothing if the recording can't be found.
     */
    set_playing(recording_id, value) {
        if (!this.#handle) {
            throw new Error(`attempted to set play state to ${value ? "playing" : "paused"} in a stopped web viewer`);
        }
        this.#handle.set_playing(recording_id, value);
    }
    /**
     * Get the current time.
     *
     * The interpretation of time depends on what kind of timeline it is:
     *
     * - For time timelines, this is the time in nanoseconds.
     * - For sequence timelines, this is the sequence number.
     *
     * This always returns `0` if the recording or timeline can't be found.
     */
    get_current_time(recording_id, timeline) {
        if (!this.#handle) {
            throw new Error(`attempted to get current time in a stopped web viewer`);
        }
        return this.#handle.get_time_for_timeline(recording_id, timeline) || 0;
    }
    /**
     * Set the current time.
     *
     * Equivalent to clicking on the timeline in the time panel at the specified `time`.
     * The interpretation of `time` depends on what kind of timeline it is:
     *
     * - For time timelines, this is the time in nanoseconds.
     * - For sequence timelines, this is the sequence number.
     *
     * This does nothing if the recording or timeline can't be found.
     */
    set_current_time(recording_id, timeline, time) {
        if (!this.#handle) {
            throw new Error(`attempted to set current time to ${time} in a stopped web viewer`);
        }
        this.#handle.set_time_for_timeline(recording_id, timeline, time);
    }
    /**
     * Get the active timeline.
     *
     * This always returns `null` if the recording can't be found.
     */
    get_active_timeline(recording_id) {
        if (!this.#handle) {
            throw new Error(`attempted to get active timeline in a stopped web viewer`);
        }
        return this.#handle.get_active_timeline(recording_id) ?? null;
    }
    /**
     * Set the active timeline.
     *
     * This does nothing if the recording or timeline can't be found.
     */
    set_active_timeline(recording_id, timeline) {
        if (!this.#handle) {
            throw new Error(`attempted to set active timeline to ${timeline} in a stopped web viewer`);
        }
        this.#handle.set_active_timeline(recording_id, timeline);
    }
    /**
     * Get the time range for a timeline.
     *
     * This always returns `null` if the recording or timeline can't be found.
     */
    get_time_range(recording_id, timeline) {
        if (!this.#handle) {
            throw new Error(`attempted to get time range in a stopped web viewer`);
        }
        return this.#handle.get_timeline_time_range(recording_id, timeline);
    }
    /**
     * Toggle fullscreen mode.
     *
     * This does nothing if `allow_fullscreen` was not set to `true` when starting the viewer.
     *
     * Fullscreen mode works by updating the underlying `<canvas>` element's `style`:
     * - `position` to `fixed`
     * - width/height/top/left to cover the entire viewport
     *
     * When fullscreen mode is toggled off, the style is restored to its previous values.
     *
     * When fullscreen mode is toggled on, any other instance of the viewer on the page
     * which is already in fullscreen mode is toggled off. This means that it doesn't
     * have to be tracked manually.
     *
     * This functionality can also be directly accessed in the viewer:
     * - The maximize/minimize top panel button
     * - The `Toggle fullscreen` UI command (accessible via the command palette, CTRL+P)
     */
    toggle_fullscreen() {
        if (!this.#allow_fullscreen)
            return;
        if (!this.#handle || !this.#canvas) {
            throw new Error(`attempted to toggle fullscreen mode in a stopped web viewer`);
        }
        if (this.#fullscreen) {
            this.#minimize();
        }
        else {
            this.#maximize();
        }
    }
    set_credentials(access_token, email) {
        if (!this.#handle) {
            throw new Error(`attempted to set credentials in a stopped web viewer`);
        }
        this.#handle.set_credentials(access_token, email);
    }
    #minimize = () => { };
    #maximize = () => {
        _minimize_current_fullscreen_viewer?.();
        const canvas = this.#canvas;
        const rect = canvas.getBoundingClientRect();
        const sync_style_to_rect = () => {
            canvas.style.left = rect.left + "px";
            canvas.style.top = rect.top + "px";
            canvas.style.width = rect.width + "px";
            canvas.style.height = rect.height + "px";
        };
        const undo_style = () => canvas.removeAttribute("style");
        const transition = (callback) => setTimeout(() => requestAnimationFrame(callback), transition_delay_ms);
        canvas.classList.add(classes.fullscreen_base, classes.fullscreen_rect);
        sync_style_to_rect();
        requestAnimationFrame(() => {
            if (!this.#fullscreen)
                return;
            canvas.classList.add(classes.transition);
            transition(() => {
                if (!this.#fullscreen)
                    return;
                undo_style();
                document.body.classList.add(classes.hide_scrollbars);
                document.documentElement.classList.add(classes.hide_scrollbars);
                this.#dispatch_event("fullscreen", true);
            });
        });
        this.#minimize = () => {
            document.body.classList.remove(classes.hide_scrollbars);
            document.documentElement.classList.remove(classes.hide_scrollbars);
            sync_style_to_rect();
            canvas.classList.remove(classes.fullscreen_rect);
            transition(() => {
                if (this.#fullscreen)
                    return;
                undo_style();
                canvas.classList.remove(classes.fullscreen_base, classes.transition);
            });
            _minimize_current_fullscreen_viewer = null;
            this.#fullscreen = false;
            this.#dispatch_event("fullscreen", false);
        };
        _minimize_current_fullscreen_viewer = () => this.#minimize();
        this.#fullscreen = true;
    };
}
export class LogChannel {
    #on_send;
    #on_send_table;
    #on_close;
    #get_state;
    #closed = false;
    /**
     * @param on_send
     * @param on_close
     * @param get_state
     */
    constructor(on_send, on_send_table, on_close, get_state) {
        this.#on_send = on_send;
        this.#on_send_table = on_send_table;
        this.#on_close = on_close;
        this.#get_state = get_state;
    }
    get ready() {
        return !this.#closed && this.#get_state() === "ready";
    }
    /**
     * Send an `rrd` containing log messages to the viewer.
     *
     * Does nothing if `!this.ready`.
     *
     * @param rrd_bytes Is an rrd file stored in a byte array, received via some other side channel.
     */
    send_rrd(rrd_bytes) {
        if (!this.ready)
            return;
        this.#on_send(rrd_bytes);
    }
    send_table(table_bytes) {
        if (!this.ready)
            return;
        this.#on_send_table(table_bytes);
    }
    /**
     * Close the channel.
     *
     * Does nothing if `!this.ready`.
     */
    close() {
        if (!this.ready)
            return;
        this.#on_close();
        this.#closed = true;
    }
}
const classes = {
    hide_scrollbars: "rerun-viewer-hide-scrollbars",
    fullscreen_base: "rerun-viewer-fullscreen-base",
    fullscreen_rect: "rerun-viewer-fullscreen-rect",
    transition: "rerun-viewer-transition",
};
const transition_delay_ms = 100;
const css = `
  html.${classes.hide_scrollbars},
  body.${classes.hide_scrollbars} {
    scrollbar-gutter: auto !important;
    overflow: hidden !important;
  }

  .${classes.fullscreen_base} {
    position: fixed;
    z-index: 99999;
  }

  .${classes.transition} {
    transition: all ${transition_delay_ms / 1000}s linear;
  }

  .${classes.fullscreen_rect} {
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
  }
`;
function injectStyle() {
    const ID = "__rerun_viewer_style";
    if (document.getElementById(ID)) {
        // already injected
        return;
    }
    const style = document.createElement("style");
    style.id = ID;
    style.appendChild(document.createTextNode(css));
    document.head.appendChild(style);
}
function setupGlobalEventListeners() {
    window.addEventListener("keyup", (e) => {
        if (e.code === "Escape") {
            _minimize_current_fullscreen_viewer?.();
        }
    });
}
//# sourceMappingURL=index.js.map