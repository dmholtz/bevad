import os
import subprocess

from bevad_sim.__about__ import __version__


def get_vcs_version() -> str:
    """
    returns a version string from the version control system which contains possible development progress:
      - "X.X.X" if we are on a release
      - "X.X.X-numCommitsOnTop-gCommitHash" if we are in an git-repo and diverged from last release
      - if VCS is unavailable use the version from __about__.py
    """
    # note: "hatch-vcs" provides a similar solution right in the build-backend. But it comes with some downsides:
    # - yet another dependency
    # - still no bullet-proof solution, as the version is only updated upon build/install and needs workarounds
    #   to obtain this, see <https://github.com/maresb/hatch-vcs-footgun-example>
    # - when e2e-core is used as editable install in other projects and these build docker-images, the full
    #   .git folder needs to be mounted (-> cache misses) and possibly the source folder must be writable
    # ==> use own simple git-call-solution
    try:
        # we need to force git to only look into e2e-core's .git folder, otherwise, if e2e-core is installed
        # from wheel (without .git), git would pick up the outer .git folder from the user project.
        bevad_sim_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        args = ["git", f"--git-dir={bevad_sim_root}/.git", "describe", "--tags", "--dirty", "--broken", "--always"]
        completed_process = subprocess.run(args, cwd=bevad_sim_root, capture_output=True, text=True, check=True)
        version = completed_process.stdout.strip()
        return version
    except (EnvironmentError, subprocess.CalledProcessError) as e:
        return __version__
