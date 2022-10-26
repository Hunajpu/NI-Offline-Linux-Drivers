#!/usr/bin/env python3
"""Create an offline repository for RPM-based systems.

Specifically, this script aims to create repositories on Red Hat Enterprise
Linux (RHEL) 7 and 8, CentOS 7 and 8, and openSUSE Leap 15.2. However, this
script should work for any system that uses the DNF, Yum, or Zypper package
managers. Other package manager support can be added by subclassing the
`PackagesDownload` class; see the existing package manager classes for examples.

This script does little work itself; most of the necessary work to create the
repositories is done by the underlying package managers and the repo creation
tool `createrepo`. Several parameters can be passed to this script to instruct
the underlying package manager to include or exclude extra packages.

By default, this script will include installed packages into the generated
repository. Only RHEL 8+ supports including installed packages when excluding
system packages; on other systems, passing the `--exclude-system` flag will
exclude both installed packages and dependent system packages (such as the
kernel package)."""

import argparse
import io
import logging
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
)

_moduleLogger = logging.getLogger(__name__)
_moduleLogger.addHandler(logging.NullHandler())

ERROR_PREVIEW_LENGTH = 512


def _main(args: argparse.Namespace) -> int:
    downloader: DownloadOperation = get_first_existing(args.operation)
    repo_dirs = args.operation.download_targets(downloader, **vars(args))
    if repo_dirs:
        createrepo_tool = SubprocessWrapper("createrepo", "--workers=5")

        for repo_dir in repo_dirs:
            _moduleLogger.info("Creating repo in '%s'", str(repo_dir))
            createrepo_tool.execute([str(repo_dir)])

    return 0


def get_first_existing(operation: Type["DownloadOperation"]) -> Any:
    """Iterate through the implementations of the specified operation, finding the first that can
    be successfully initialized.
    """
    programs = operation.__subclasses__()
    for program in programs:
        try:
            return program()  # type: ignore
        except (FileNotFoundError, TypeError):
            pass

    raise FileNotFoundError(f"No program found in {[p.__name__ for p in programs]}")


class SubprocessWrapper:  # pylint: disable=too-few-public-methods
    """Encapsulates logic to find the programs and capture their output to the module logger."""

    path: Path

    def __init__(self, name: str, *args: str):
        program_path = shutil.which(name)
        if program_path is None:
            raise FileNotFoundError(f"Could not find program '{name}'")
        self.path = Path(program_path)
        self.global_args = list(args)

    @staticmethod
    def _read_data(
        stream: io.BufferedIOBase, logger: Callable[[str], None]
    ) -> bytearray:
        data = bytearray()
        while True:
            chunk = stream.read1(select.PIPE_BUF)
            if not chunk:
                break

            data.extend(chunk)
            logger(chunk.decode())

        return data

    def execute(
        self,
        cmds: Optional[List[str]] = None,
        cwd: Optional[Path] = None,
        stderr_logger: Callable[..., None] = _moduleLogger.error,
    ) -> Tuple[int, Tuple[bytes, bytes]]:
        """Run this process, capturing and logging output."""
        if cmds is None:
            cmds = []

        cmds = [str(self.path)] + self.global_args + cmds
        cmd_string = " ".join(cmds)

        _moduleLogger.debug("Executing '%s'", cmd_string)
        with subprocess.Popen(
            cmds,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as process:
            # This is two consecutive processing loops, processing stdout first with the logic that
            # stderr will be empty if there's no error and stdout will probably not output much
            # data if stderr has reported an error. This is vastly simpler than trying to process
            # both streams at the same time and usually provides the same result.
            data = (
                self._read_data(process.stdout, _moduleLogger.debug),  # type: ignore
                self._read_data(process.stderr, stderr_logger),  # type: ignore
            )

        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, process.args, *map(bytearray.decode, data)
            )

        return process.returncode, data


class DownloadOperation:
    """Base class for download operations."""

    program: SubprocessWrapper

    def __init__(self, program: SubprocessWrapper):
        self.program = program

    @classmethod
    def process_args(
        cls, parser: argparse.ArgumentParser, args: argparse.Namespace
    ) -> None:
        """Perform any additional argument parsing associated with this download operation."""
        raise NotImplementedError()

    def download_targets(
        self,
        targets: List[str],
        output: Path,
        *args: Any,
        **kwargs: Any,
    ) -> Iterable[Path]:
        """Download the specified targets to the specified path."""
        raise NotImplementedError()

    @staticmethod
    def _begin_download(
        message_format: str,
        targets: List[str],
        output: Path,
        should_exit: Callable[[], bool],
        *_args: Any,
        **_kwargs: Any,
    ) -> bool:
        _moduleLogger.warning(
            message_format,
            len(targets),
            str(output),
        )

        if should_exit():
            _moduleLogger.info("Exiting on user command")
            return False

        return True


class PackagesDownload(DownloadOperation):
    """Download operation using package managers to download packages and their dependecies."""

    class Config(NamedTuple):
        """Configuration specific to Packages Download operations."""

        exclude_installed: bool
        exclude_recommends: bool
        exclude_system: bool
        known_repos: Dict[str, Dict[str, str]]
        enabled_repos: Dict[str, Dict[str, str]]

    @classmethod
    def process_args(cls, parser: argparse.ArgumentParser, args: argparse.Namespace):
        args.exclude_installed &= args.exclude_system
        if args.enabled_repos:
            args.enabled_repos = set(args.enabled_repos)

    def clean(self, *args: str):
        """Clean the package cache."""
        _moduleLogger.info("Cleaning package cache")
        cmd_args = ["clean"]
        if args:
            cmd_args.extend(args)
        self.program.execute(cmd_args)

    def download_packages(
        self,
        packages: List[str],
        output: Path,
        config: Config,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Download the specified packages with the specified configuration."""
        raise NotImplementedError()

    def download_targets(
        self,
        targets: List[str],
        output: Path,
        *args: Any,
        **kwargs: Any,
    ) -> Iterable[Path]:
        if not self._begin_download(
            "Downloading %d package(s) and dependencies to '%s'",
            targets,
            output,
            *args,
            **kwargs,
        ):
            return []

        config = self._get_config_from_args(*args, **kwargs)
        self.download_packages(targets, output, config, *args, **kwargs)
        return (output,)

    def get_known_repos(self) -> Dict[str, Dict[str, str]]:
        """Get the known enabled repos for this program."""
        raise NotImplementedError()

    def refresh(self, *args: str):
        """Refresh the feeds on the system."""
        _moduleLogger.info("Refreshing repo metadata")
        self.program.execute(list(args))

    def _get_config_from_args(
        self,
        exclude_installed: bool,
        exclude_recommends: bool,
        exclude_system: bool,
        enabled_repos: List[str],
        *_args: Any,
        **_kwargs: Any,
    ) -> Config:
        if enabled_repos:
            known_repos = self.get_known_repos()
            try:
                repos = {repo: known_repos[repo] for repo in enabled_repos}
            except KeyError:
                unknown_repos = [
                    repo for repo in enabled_repos if repo not in known_repos
                ]
                raise ValueError(  # pylint: disable=raise-missing-from
                    f"The following repo-ids are unknown: {unknown_repos}"
                )
        else:
            known_repos = {}
            repos = {}

        return self.Config(
            exclude_installed,
            exclude_recommends,
            exclude_system,
            known_repos,
            repos,
        )


class RepoDownload(DownloadOperation):
    """Operation to clone entire repositories."""

    @classmethod
    def process_args(cls, parser: argparse.ArgumentParser, args: argparse.Namespace):
        if len(args.targets) > 1 and args.no_repo_path:
            parser.error("Can't specify multiple repos and '--norepopath'")

    def clone_repos(
        self,
        repos: Iterable[str],
        output: Path,
        no_repo_path: bool,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Download the specified repos to the specified path."""
        raise NotImplementedError()

    def download_targets(
        self,
        targets: List[str],
        output: Path,
        *args: Any,
        **kwargs: Any,
    ) -> Iterable[Path]:
        if not self._begin_download(
            "Cloning %d repo(s) to '%s'",
            targets,
            output,
            *args,
            **kwargs,
        ):
            return []

        self.clone_repos(targets, output, *args, **kwargs)
        return (
            (output,)
            if kwargs["no_repo_path"]
            else filter(Path.is_dir, output.iterdir())
        )


class Dnf(PackagesDownload, RepoDownload):
    """Wrapper around the `DNF` package manager on RHEL 8."""

    def __init__(self):
        super().__init__(SubprocessWrapper("dnf", "--assumeyes"))

    def clone_repos(
        self,
        repos: Iterable[str],
        output: Path,
        no_repo_path: bool,
        *_args: Any,
        **_kwargs: Any,
    ):
        cmds = ["reposync", "-p", str(output)] + [f"--repoid={repo}" for repo in repos]
        if no_repo_path:
            cmds.append("--norepopath")

        _moduleLogger.info("Starting repo clone")
        self.program.execute(cmds)

    def download_packages(
        self,
        packages: List[str],
        output: Path,
        config: PackagesDownload.Config,
        *_any: Any,
        **_kwargs: Any,
    ):
        self.clean("packages")
        self.refresh()

        cmds = [
            "download",
            "--resolve",
            "--downloadonly",
            "--downloaddir",
            str(output),
        ]

        if not config.exclude_installed:
            cmds.append("--alldeps")

        repo_cmds: List[str] = []
        for repo_id in config.enabled_repos:
            repo_cmds.extend(("--repo", repo_id))

        cmds = repo_cmds + cmds

        with TemporaryDirectory() as tempdir:
            if not config.exclude_system:
                cmds = ["--installroot", tempdir, "--releasever=/"] + cmds

            _moduleLogger.info("Starting package download")
            self.program.execute(cmds + packages)

    def get_known_repos(self) -> Dict[str, Dict[str, str]]:
        _, (stdout, _) = self.program.execute(["repolist"])
        lines = stdout.decode().splitlines()[3:]
        return dict(map(self._parse_repo, lines))

    def refresh(self, *args: str):
        return super().refresh("makecache", *args)

    @staticmethod
    def _parse_repo(string: str) -> Tuple[str, Dict[str, str]]:
        """
        >>> Dnf._parse_repo("test-repo\tTest Repo Name")
        ('test-repo', {'Name': 'Test Repo Name'})
        """
        raw_id, raw_name = string.split(maxsplit=1)
        return (raw_id.strip(), {"Name": raw_name.strip()})


class Yum(PackagesDownload):
    """Wrapper around the `Yum` package manager on RHEL 7."""

    def __init__(self):
        super().__init__(SubprocessWrapper("yum", "--assumeyes"))

    def download_packages(
        self,
        packages: List[str],
        output: Path,
        config: PackagesDownload.Config,
        *_any: Any,
        **_kwargs: Any,
    ):
        self.clean("packages")
        self.refresh()

        cmds = [
            "install",
            "--downloadonly",
            "--downloaddir",
            str(output),
        ]

        extra_repos = config.known_repos.keys() - config.enabled_repos.keys()
        for repo_id in extra_repos:
            cmds.insert(0, f"--disablerepo={repo_id}")

        with TemporaryDirectory() as tempdir:
            if not config.exclude_system:
                cmds = ["--installroot", tempdir, "--releasever=/"] + cmds

            _moduleLogger.info("Starting package download")
            self.program.execute(cmds + packages)

    def get_known_repos(self) -> Dict[str, Dict[str, str]]:
        _, (stdout, _) = self.program.execute(["repolist"])
        lines = stdout.decode().splitlines()[3:]
        return dict(map(self._parse_repo, lines))

    def refresh(self, *args: str):
        return super().refresh("makecache", *args)

    @staticmethod
    def _parse_repo(string: str) -> Tuple[str, Dict[str, str]]:
        """
        >>> Yum._parse_repo("test-repo\tTest Repo Name")
        ('test-repo', {'Name': 'Test Repo Name'})

        >>> Yum._parse_repo("test-repo/extra_info\tTest Repo Name")
        ('test-repo', {'Name': 'Test Repo Name'})
        """
        raw_id, raw_name = string.split(maxsplit=1)

        # built-in repos have extra info separated by forward slashes that prevent repo-id matching
        repo_id = raw_id.split("/", maxsplit=1)[0]
        return (repo_id.strip(), {"Name": raw_name.strip()})


class Zypper(PackagesDownload):
    """Wrapper around the `Zypper` package manager on openSUSE."""

    DOWNLOAD_ROOT = Path("/var/cache/zypp/packages")

    def __init__(self):
        super().__init__(SubprocessWrapper("zypper", "--non-interactive"))

    @staticmethod
    def _copy_packages_to_output(output: Path):
        count = 0
        for count, rpm in enumerate(Zypper.DOWNLOAD_ROOT.rglob("*.rpm"), start=1):
            _moduleLogger.debug("(%d) %s", count, rpm.name)
            shutil.copy(rpm, output)

        _moduleLogger.info("Copied %d rpms to '%s'", count, str(output))

    def download_packages(
        self,
        packages: List[str],
        output: Path,
        config: PackagesDownload.Config,
        *_any: Any,
        **_kwargs: Any,
    ):
        self.clean()
        self.refresh()

        cmds = [
            "--gpg-auto-import-keys",
            "install",
            "--download-only",
            "--no-recommends" if config.exclude_recommends else "--recommends",
        ]

        for repo_id in config.enabled_repos:
            cmds.extend(("--repo", repo_id))

        if not config.exclude_system:
            cmds.insert(0, "--disable-system-resolvables")

        _moduleLogger.info("Starting package download")
        self.program.execute(cmds + packages)

        self._copy_packages_to_output(output)

    def get_known_repos(self) -> Dict[str, Dict[str, str]]:
        cmds = ["repos", "--show-enabled-only", "--url"]
        _, (stdout, _) = self.program.execute(cmds)
        lines = stdout.decode().splitlines()[4:]
        return dict(map(self._parse_repo, lines))

    def refresh(self, *args: str):
        return super().refresh("refresh", *args)

    @staticmethod
    def _parse_repo(string: str) -> Tuple[str, Dict[str, str]]:
        """
        >>> Zypper._parse_repo("1 | test-repo | Test Repo Name | Yes | (r ) Yes | No | https://url.com")
        ('test-repo', {'Name': 'Test Repo Name', 'URL': 'https://url.com'})
        """
        _, repo_id, raw_name, _, _, _, raw_url = string.split("|", maxsplit=6)
        return (repo_id.strip(), {"Name": raw_name.strip(), "URL": raw_url.strip()})


class RepoSync(RepoDownload):
    """Wrapper around the `reposync` tool on RHEL 7."""

    def __init__(self):
        super().__init__(SubprocessWrapper("reposync"))

    def clone_repos(
        self,
        repos: Iterable[str],
        output: Path,
        no_repo_path: bool,
        *_args: Any,
        **_kwargs: Any,
    ):
        cmds = ["--download_path", str(output)] + [f"--repoid={repo}" for repo in repos]
        if no_repo_path:
            cmds.append("--norepopath")

        _moduleLogger.info("Starting repo clone")
        self.program.execute(cmds)


class Wget(RepoDownload):
    """Wrapper around the `wget` tool."""

    zypper: Zypper

    def __init__(self):
        super().__init__(
            SubprocessWrapper(
                "wget",
                "--accept",
                "rpm",
                "--no-directories",
                "--execute",
                "robots=off",
                "--mirror",
                "--no-parent",
            )
        )
        self.zypper = Zypper()

    def clone_repos(
        self,
        repos: Iterable[str],
        output: Path,
        no_repo_path: bool,
        *_args: Any,
        **_kwargs: Any,
    ):
        known_repos = self.zypper.get_known_repos()
        for repo in repos:
            if repo not in known_repos:
                raise ValueError(f"'{repo}' is not a known repo")

            repo_path = output
            if not no_repo_path:
                repo_path = output.joinpath(repo)

            repo_path.mkdir(exist_ok=True)
            cmds = ["--directory-prefix", str(repo_path), known_repos[repo]["URL"]]

            _moduleLogger.info("Starting repo clone")
            self.program.execute(cmds)


def get_parser() -> argparse.ArgumentParser:
    """Get the argument parser for this script."""
    short_description, _, long_description = __doc__.split("\n", 2)  # type: ignore
    examples = """

Examples:

    Create a repository for the GPIB driver and its dependencies
        %(prog)s packages ni-488.2

    Create a repository containing both the NI-VISA and NI-Serial drivers and their
    dependencies in a directory named "my_repo"
        %(prog)s --output my_repo packages ni-visa ni-serial

    Clone the ni-software-2022 repository with max verbosity logging
        %(prog)s -vv repo ni-software-2022

"""
    parser = argparse.ArgumentParser(
        description=short_description,
        epilog=long_description + examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output",
        default=Path("offline_repo"),
        type=Path,
        help="output path",
    )
    parser.add_argument(
        "-y", "--assumeyes", action="store_true", help="answer yes to all questions"
    )

    debug_group = parser.add_argument_group("debug arguments")
    debug_group.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="don't do anything, just report what would be done",
    )
    debug_group.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="print debug information (repeat for more detailed output)",
    )
    debug_group.add_argument(
        "-q",
        "--quiet",
        action="count",
        default=0,
        help="print only essential information (repeat for quieter output)",
    )

    subparsers = parser.add_subparsers(help="run `{subcommand} --help` for more help")

    packages_parser = subparsers.add_parser(
        "packages", help="create a new repository based on package dependencies"
    )
    packages_parser.add_argument(
        "targets",
        metavar="PACKAGE",
        nargs="+",
        help="target packages to define the repository",
    )
    packages_parser.add_argument(
        "--exclude-system",
        action="store_true",
        help="exclude installed and system packages",
    )
    packages_parser.add_argument(
        "-r",
        "--repo",
        action="append",
        dest="enabled_repos",
        metavar="REPO",
        help="use packages only from the specified repository (may be specified multiple times)",
    )

    distro_group = packages_parser.add_argument_group("distro-specific arguments")
    distro_group.add_argument(
        "--exclude-installed",
        action="store_true",
        help="exclude installed packages (only supported on RHEL 8+)",
    )
    distro_group.add_argument(
        "--exclude-recommends",
        action="store_true",
        help="exclude recommended dependencies (only supported on openSUSE systems)",
    )

    packages_parser.set_defaults(operation=PackagesDownload)

    repo_parser = subparsers.add_parser("repo", help="clone existing repositories")
    repo_parser.add_argument(
        "targets", metavar="REPO", nargs="+", help="repositories to clone"
    )
    repo_parser.add_argument(
        "--norepopath",
        dest="no_repo_path",
        action="store_true",
        help=(
            "clone to the output directory directly opposed to a subdirectory with the repository name"
            + " (only when specifying one repository)"
        ),
    )
    repo_parser.set_defaults(operation=RepoDownload)

    return parser


def parse_args(parser: argparse.ArgumentParser, argv: List[str]) -> argparse.Namespace:
    """Parse the specified arguments with the specified parser."""
    args = parser.parse_args(argv)

    # We want to default to WARNING
    # Verbosity gives us granularity to control past that
    if args.verbose > 0 and args.quiet > 0:
        parser.error("Mixing --verbose and --quiet is contradictory")
    verbosity = 2 + args.quiet - args.verbose
    verbosity = max(verbosity, 0)
    verbosity = min(verbosity, 4)
    args.logging_level = {
        0: logging.DEBUG,
        1: logging.INFO,
        2: logging.WARNING,
        3: logging.ERROR,
        4: logging.CRITICAL,
    }[verbosity]
    args.logging_level_name = logging.getLevelName(args.logging_level)
    args.logging_format = "(%(asctime)s) %(levelname)-5s : %(message)s"

    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    args.operation.process_args(parser, args)

    _should_exit = False if args.assumeyes else None
    if args.dry_run or (args.logging_level > logging.WARNING and _should_exit is None):
        _should_exit = True

    def should_exit() -> bool:
        if _should_exit is not None:
            return _should_exit

        while True:
            response = input("Is this ok? [y/N]: ")

            if not response or response.lower() == "n":
                return True

            if response.lower() == "y":
                return False

    args.should_exit = should_exit

    return args


def assert_root(args: argparse.Namespace) -> int:
    """Restarts the script as root if not running as root."""
    uid = os.geteuid()
    _moduleLogger.debug("Running as UID %d", uid)

    if uid != 0:
        _moduleLogger.warning("This script is not being run as root!")
        _moduleLogger.info("Invoking `sudo` to run as root")
        # We've already checked the returncode in the child process. No need to double check.
        process = subprocess.run(  # pylint: disable=subprocess-run-check
            ["sudo", sys.executable] + sys.argv
        )
        return process.returncode

    return _main(args)


def log_error_preview(error: subprocess.CalledProcessError):
    """Log a preview of the called process error."""
    # Try to print stderr first. If it's empty, switch to stdout
    message = (error.stderr or error.stdout)[-ERROR_PREVIEW_LENGTH:]
    if len(message) == ERROR_PREVIEW_LENGTH:
        message = "..." + message[3:]


def main(argv: Optional[List[str]] = None):
    """Entry point for the application."""
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(get_parser(), argv)

    logging.basicConfig(
        level=args.logging_level_name,
        format=args.logging_format,
    )

    # assert root after setting up logging to see messages
    exit_code = 255
    try:
        exit_code = assert_root(args)
    except Exception as err:  # pylint: disable=broad-except
        # print stacktrace on detailed logging
        logger = (
            _moduleLogger.error
            if args.logging_level > logging.DEBUG
            else _moduleLogger.exception
        )
        logger("The program raised an exception.")

        if isinstance(err, subprocess.CalledProcessError):
            # pass exit code up on subprocess error
            exit_code = err.returncode

            if args.logging_level == logging.INFO:
                log_error_preview(err)

        if args.logging_level > logging.DEBUG:
            _moduleLogger.error(
                "Rerun with a higher verbosity (-v) to see more information."
            )
    finally:
        _moduleLogger.info("Exiting with code %d", exit_code)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
