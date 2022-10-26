#!/usr/bin/env python3
"""Create an offline repository for Debian-based systems.

Specifically, this script aims to create repositories on Ubuntu 20.04. However,
this script should work for any system that uses the APT package manager. Other
package manager support can be added by subclassing the `PackagesDownload`
class; see the existing package manager classes for examples.

This script does little work itself; most of the necessary work to create the
repositories is done by the underlying package manager and repository creation
tool `dpkg-scanpackages`. Several parameters can be passed to this script to
instruct the underlying package manager to include or exclude other packages. By
default, this script will include installed packages into the generated
repository. Pass the `--exclude-system` flag to exclude both installed and
dependent system packages (such as the kernel package).

Additionally, this script can clone entire repositories if the `apt-mirror` tool
is installed on the system. The targets for this `repo` command can either be
known repository names or Apt source entries. A repository name is known if the
`/etc/apt/sources.list.d/` directory contains a file with the name
`{name}.list`. An Apt source entry is a string that specifies where an Apt
repository can be found and generally follows the pattern
`deb http://site.example.com/debian distribution component1 component2`. You can
see examples of this format in the `/etc/apt/sources.list` file or read about it
on the Debian wiki: https://wiki.debian.org/SourcesList#sources.list_format.
"""

import argparse
import contextlib
from datetime import datetime, timezone
from getpass import getpass
import gzip
import hashlib
from importlib import util as import_utils
import io
import itertools
import logging
import os
from pathlib import Path
import re
import select
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Sized,
    Tuple,
    Type,
    Union,
)

python_gnupg_installed = import_utils.find_spec("gnupg") is not None
if python_gnupg_installed:
    import gnupg

    try:
        getattr(gnupg.GPG, "sign_file")
    except AttributeError as err:
        raise RuntimeError(
            "Module 'gnupg' exists, but does not contain the expected utilities."
            + " Ensure pip package `python-gnupg` is installed, not `gnupg`."
        ) from err

_moduleLogger = logging.getLogger(__name__)
_moduleLogger.addHandler(logging.NullHandler())

ERROR_PREVIEW_LENGTH = 512


def _main(args: argparse.Namespace):
    downloader: DownloadOperation = get_first_existing(args.operation)
    vargs = vars(args)
    args.operation.download_targets(downloader, **vargs)
    # Clone is only possible with apt-mirror, which will have the necessary repo data already
    if issubclass(args.operation, PackagesDownload):
        createrepo_tool = DpkgScanPackages()

        _moduleLogger.info("Creating repo in '%s'", args.output)
        createrepo_tool.create_repo(**vargs)

        if not args.no_sign:
            _moduleLogger.info("Signing repo at '%s'", args.output)
            metadata_path = create_repo_metadata(**vargs)
            sign_repo_metadata(metadata_path, **vargs)

    _moduleLogger.info("Done!")
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
        targets: Sequence[str],
        output: Path,
        *args: Any,
        **kwargs: Any,
    ) -> Iterable[Path]:
        """Download the specified targets to the specified path."""
        raise NotImplementedError()

    @staticmethod
    def _begin_download(
        message_format: str,
        targets: Sized,
        output: Path,
        should_exit: Callable[[], bool],
        *_args: Any,
        **_kwargs: Any,
    ) -> bool:
        _moduleLogger.warning(message_format, len(targets), str(output))

        if should_exit():
            _moduleLogger.info("Exiting on user command")
            return False

        return True


class PackagesDownload(DownloadOperation):
    """Download operation using package managers to download packages and their dependecies."""

    @classmethod
    def process_args(cls, parser: argparse.ArgumentParser, args: argparse.Namespace):
        if not args.architectures:
            args.architectures = ["amd64", "i386"]

        if not args.no_sign:
            if not python_gnupg_installed:
                parser.error(
                    "`--no-sign` was not passed, but required module `gnupg` could not be loaded."
                    + " Run `pip install python-gnupg` to allow signing the created repository "
                    + "or pass `--no-sign` to prevent signing."
                )

            if args.passphrase:
                if args.passphrase == "-":
                    if sys.stdin.isatty():
                        _moduleLogger.debug("Waiting on user input")
                        args.passphrase = getpass("GPG Key Passphrase: ")
                    else:
                        _moduleLogger.warning("Reading GPG key passphrase from stdin")
                        args.passphrase = sys.stdin.readline().strip()
                else:
                    filepath = Path(args.passphrase)
                    if filepath.exists() and filepath.is_file():
                        with filepath.open("r", encoding="utf-8") as file_:
                            args.passphrase = file_.read().strip()

            if args.assumeyes and (not args.no_passphrase and args.passphrase is None):
                parser.error(
                    "\n".join(
                        (
                            "`--assumeyes` was passed without `--no-sign`, but passphrase was not specified.",
                            "Solutions:",
                            " 1. Pass `--passphrase` with either:",
                            "   a. the passphrase",
                            "   b. the path to a file containing the passphrase",
                            "   c. `-` to read the passphrase from stdin",
                            " 2. Pass `--no-passphrase` to specify no passphrase is required",
                            " 3. Pass `--no-sign` to prevent signing.",
                        )
                    )
                )

    def clean(self, *args: str):
        """Clean the package cache."""
        _moduleLogger.info("Cleaning package cache")
        cmd_args = ["clean"]
        if args:
            cmd_args.extend(args)
        self.program.execute(cmd_args)

    def download_packages(
        self,
        packages: Sequence[str],
        output: Path,
        config: "Config",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Download the specified packages with the specified configuration."""
        raise NotImplementedError()

    def download_targets(
        self,
        targets: Sequence[str],
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

        config = self.Config.from_args(*args, **kwargs)
        self.download_packages(targets, output, config, *args, **kwargs)
        return (output,)

    def refresh(self, *args: str):
        """Refresh the feeds on the system."""
        _moduleLogger.info("Refreshing repo metadata")
        self.program.execute(list(args))

    class Config(NamedTuple):
        """Configuration specific to Packages Download operations."""

        component: str
        exclude_recommends: bool
        exclude_suggests: bool
        exclude_system: bool
        sources_list: Optional[Path]

        @classmethod
        def from_args(
            cls,
            component: str,
            exclude_recommends: bool,
            exclude_suggests: bool,
            exclude_system: bool,
            sources_list: Optional[Path],
            *_args: Any,
            **_kwargs: Any,
        ):
            """Instantiate Config from arguments."""
            return cls(
                component,
                exclude_recommends,
                exclude_suggests,
                exclude_system,
                sources_list,
            )


class RepoDownload(DownloadOperation):
    """Operation to clone entire repositories."""

    @classmethod
    def process_args(cls, parser: argparse.ArgumentParser, args: argparse.Namespace):
        if not args.architectures:
            args.architectures = ["amd64", "i386"]

    def clone_repos(
        self,
        targets: Sequence[str],
        output: Path,
        collapse: bool,
        architectures: Iterable[str],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Download the specified repos to the specified path."""
        raise NotImplementedError()

    def download_targets(
        self,
        targets: Sequence[str],
        output: Path,
        *args: Any,
        **kwargs: Any,
    ) -> Iterable[Path]:
        if self._begin_download(
            "Cloning %d repo(s) to '%s'",
            targets,
            output,
            *args,
            **kwargs,
        ):
            self.clone_repos(targets, output, *args, **kwargs)  # type: ignore

        return (output,)


class AptWrapper(PackagesDownload):
    """Wrapper around the 'apt' package manager on Ubuntu"""

    def __init__(self):
        super().__init__(SubprocessWrapper("apt", "--yes"))

    @staticmethod
    def add_package_to_pool(pool_path: Path, package: Path):
        """Move the specified `package` to the proper place inside the specified `pool_path`."""
        base_name = package.name.split("_", maxsplit=1)[0]
        short_name = base_name[:4]
        path = pool_path.joinpath(short_name, base_name, package.name)

        _moduleLogger.debug('Moving "%s" to "%s"', package.name, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        package.rename(path)

    @staticmethod
    def create_install_root(src: Path, dst: Path):
        """Create a folder with the necessary files to run this process.

        This created folder allows us to determine dependencies isolated from the currently
        packages on the system."""
        directories = [
            "etc",
            "usr/lib",
            "var/lib/dpkg",
            "var/lib/apt/lists/partial",
            "var/cache/apt/archives/partial",
        ]
        files = ["var/lib/dpkg/status"]
        sources = ["etc/apt", "usr/lib/apt"]

        for directory in directories:
            dst.joinpath(directory).mkdir(parents=True, exist_ok=True)

        for file_ in files:
            dst.joinpath(file_).touch()

        for source in sources:
            shutil.copytree(src.joinpath(source), dst.joinpath(source))

    def download_packages(
        self,
        packages: Sequence[str],
        output: Path,
        config: PackagesDownload.Config,
        *_any: Any,
        **_kwargs: Any,
    ):
        """Download the specified packages (and their dependencies) to the specified path."""
        with TemporaryDirectory() as tempdir:
            root_path = Path("/")
            temp_path = Path(tempdir)

            if not config.exclude_system:
                self.create_install_root(root_path, temp_path)
                root_path = temp_path

            self.program.global_args.extend(("--option", f"Dir={str(root_path)}"))
            if config.sources_list:
                self.program.global_args.extend(
                    ("--option", f"Dir::Etc::SourceList={str(config.sources_list)}")
                )

            self.clean()
            self.refresh("update")

            cmds = ["install", "--download-only", "--show-progress"]

            if not config.exclude_suggests:
                cmds.append("--install-suggests")

            if config.exclude_recommends:
                cmds.append("--no-install-recommends")

            cmds.extend(packages)

            _moduleLogger.info("Starting package download")
            self.program.execute(cmds)

            self._copy_packages_to_output(config.component, root_path, output)

    def _copy_packages_to_output(self, component: str, path: Path, output_path: Path):
        cache_path = path.joinpath("var", "cache", "apt", "archives")
        deb_packages = list(cache_path.glob("*.deb"))

        _moduleLogger.info("Moving %d packages to the pool", len(deb_packages))

        pool_path = output_path.joinpath("pool", component)
        pool_path.mkdir(parents=True, exist_ok=True)
        for package in deb_packages:
            self.add_package_to_pool(pool_path, package)


class DpkgScanPackages:
    """Wrapper around the `dpkg-scanpackages` tool to create repositories on Ubuntu."""

    class Config(NamedTuple):
        """Configuration specific to the dpkg-scanpackages tool."""

        architectures: Tuple[str, ...]
        component: str
        suite: str
        metadata: Dict[str, str]

        @classmethod
        def from_args(
            cls,
            architectures: Sequence[str],
            component: str,
            suite: str,
            metadata: Mapping[str, str],
            *_args: Any,
            **_kwargs: Any,
        ):
            """Instantiate Config from arguments."""
            return cls(tuple(architectures), component, suite, dict(metadata))

    def __init__(self):
        self.program = SubprocessWrapper("dpkg-scanpackages")

    def create_repo(self, output: Path, *args: Any, **kwargs: Any):
        """Scan the specified `path`, creating a repository from the contained packages."""
        config = self.Config.from_args(*args, **kwargs)
        main_path = output.joinpath("dists", config.suite, config.component)
        main_path.mkdir(parents=True, exist_ok=True)

        for arch in config.architectures:
            data, compressed_data = self.scan_pool_for_arch(output, arch)

            arch_path = main_path.joinpath(f"binary-{arch}")
            arch_path.mkdir(exist_ok=True)
            with arch_path.joinpath("Packages").open("wb") as file_:
                file_.write(data)
            with arch_path.joinpath("Packages.gz").open("wb") as file_:
                file_.write(compressed_data)

    def scan_pool_for_arch(self, repo_path: Path, arch: str) -> Tuple[bytes, bytes]:
        """Scan the pool for packages with the specified `arch`, compiling their info."""
        _, (data, _) = self.program.execute(
            ["--arch", arch, "pool"],
            cwd=repo_path,
            stderr_logger=_moduleLogger.info,
        )
        return data, gzip.compress(data, compresslevel=9)


def create_repo_metadata(
    output: Path,
    architectures: Sequence[str],
    metadata: Mapping[str, str],
    suite: str,
    *_args: Any,
    **_kwargs: Any,
) -> Path:
    """Create the 'Release' file for the repository."""
    _moduleLogger.debug("Writing repo metadata to Release file")

    suite_path = output.joinpath("dists", suite)
    release_data = ReleaseMetadata.from_path(suite_path, architectures, metadata)
    release_path = suite_path.joinpath("Release")

    with release_path.open("w") as release_f:
        release_f.write(str(release_data))

    return release_path


def sign_repo_metadata(
    path: Path,
    key_id: Optional[str],
    passphrase: Optional[str],
    no_passphrase: bool,
    *_args: Any,
    **_kwargs: Any,
):
    """Sign the specified file with both detached and attached signatures."""
    if not no_passphrase and passphrase is None:
        passphrase = getpass("GPG Key Passphrase: ")

    sign_tool = SignTool(key_id=key_id, passphrase=passphrase)

    _moduleLogger.debug(
        'Signing repo metadata at "%s" with key "%s"', path, sign_tool.key_id
    )

    sign_tool.sign_file(path, path.with_suffix(".gpg"), detach=True)
    sign_tool.sign_file(path, path.with_name("In" + path.name))
    sign_tool.export_key(path.with_name("key.asc"))


class SignTool:
    """Wrapper for gnupg's `GPG` class to better handle failure."""

    def __init__(
        self,
        *_args: Any,
        key_id: Optional[str] = None,
        passphrase: Optional[str] = None,
        **_kwargs: Any,
    ):
        self.gpg = gnupg.GPG()
        self.key_id = self._assert_key_exists(key_id)
        self.passphrase = passphrase

    def export_key(self, output_path: Path):
        """Export the key to the specified path."""
        _moduleLogger.debug("Exporting GPG key to '%s'", output_path)
        key_text: str = self.gpg.export_keys(keyids=self.key_id, armor=True)  # type: ignore
        with output_path.open("w") as key_file:
            key_file.write(key_text)

    def sign_file(self, file_path: Path, output_path: Path, detach: bool = False):
        """Sign the specified `file_path`."""
        _moduleLogger.debug(
            "Signing file %s to create %s (detached = %s)",
            file_path,
            output_path,
            detach,
        )
        with file_path.open("rb") as file_:
            sign_result: gnupg.Sign = self.gpg.sign_file(  # type: ignore
                file_,
                keyid=self.key_id,
                passphrase=self.passphrase,
                detach=detach,
                output=str(output_path.resolve()),
                extra_args=["--digest-algo", "SHA512"],
            )

        if sign_result.status is None:  # type: ignore
            raise SignTool.SigningError(sign_result)

    def _assert_key_exists(self, key_id: Optional[str]) -> str:  # type: ignore
        """Assert the key associated with this tool exists.

        If no key was specified, asserts any key exists and is available.
        """
        _moduleLogger.debug("Asserting GPG key exists")

        keys: Optional[Sequence[str]] = self.gpg.list_keys(keys=key_id)  # type: ignore
        if keys is None:
            raise SignTool.KeyNotFoundError(key_id)

        key_id: Optional[str] = keys.curkey["keyid"]  # type: ignore
        if key_id is None:  # type: ignore
            raise SignTool.KeyNotFoundError(key_id)
        return key_id

    class KeyNotFoundError(Exception):
        """GPG key not found error."""

        def __init__(self, key_id: Optional[str] = None):
            message = (
                "No default key exists. Ensure a GPG key is available for signing"
                if key_id is None
                else f"Key '{key_id}' was not found. Ensure this key exists"
            )
            super().__init__(
                message
                + " before running this script. Note that keys are usually user-specific."
            )

    class SigningError(Exception):
        """GPG signing failed error."""

        def __init__(self, result: Any):
            assert result is gnupg.Sign
            super().__init__(result.stderr)  # type: ignore


class HashData(NamedTuple):
    """Collection of hash data for a file."""

    value: str
    size: str
    path: str

    def __str__(self) -> str:
        return " " + " ".join(self)

    @staticmethod
    def from_path(path: Path, base_path: Path) -> List["HashData"]:
        """Calculate the hash data for the file at the specified `path`."""
        _moduleLogger.debug('Getting file hashes for "%s"', str(path))
        size = path.stat().st_size
        relative_path = path.relative_to(base_path)

        with path.open("rb") as file_:
            data = file_.read()

        return [
            HashData(hasher(data).hexdigest(), str(size), str(relative_path))
            for hasher in (hashlib.md5, hashlib.sha1, hashlib.sha256, hashlib.sha512)
        ]


class ReleaseMetadata(NamedTuple):
    """Debian-based repository metadata."""

    suite: str
    architectures: Tuple[str, ...]
    components: Tuple[str, ...]
    md5_sums: Tuple[HashData, ...]
    sha1_sums: Tuple[HashData, ...]
    sha256_sums: Tuple[HashData, ...]
    sha512_sums: Tuple[HashData, ...]
    metadata: Dict[str, str]

    def __str__(self):
        return "\n".join(self.items())

    @property
    def date(self) -> str:
        """Returns the current date in the expected format."""
        time_format = "%a, %d %b %Y %H:%M:%S %Z"  # Ex: Mon, 26 Jul 2021 14:29:07 UTC
        return datetime.now(timezone.utc).strftime(time_format)

    @staticmethod
    def from_path(
        suite_path: Path, architectures: Sequence[str], metadata: Mapping[str, str]
    ) -> "ReleaseMetadata":
        """Generate the repository metadata from the specified path."""
        _moduleLogger.info("Determining suite metadata...")
        components = [path.name for path in suite_path.glob("*") if path.is_dir()]
        files = suite_path.rglob("Packages*")
        md5_sums, sha1_sums, sha256_sums, sha512_sums = zip(
            *[HashData.from_path(file_path, suite_path) for file_path in files]
        )

        return ReleaseMetadata(
            suite_path.name,
            tuple(architectures),
            tuple(components),
            md5_sums,
            sha1_sums,
            sha256_sums,
            sha512_sums,
            dict(metadata),
        )

    def items(self) -> Generator[str, None, None]:
        """Get all metadata as strings."""
        metadata = {
            "Suite": self.suite,
            "Date": self.date,
            "Architectures": " ".join(self.architectures),
            "Components": " ".join(self.components),
        }
        metadata.update(self.metadata)
        hashes = {
            "Md5Sum": self.md5_sums,
            "SHA1": self.sha1_sums,
            "SHA256": self.sha256_sums,
            "SHA512": self.sha512_sums,
        }

        yield from (f"{key}: {value}" for key, value in metadata.items())
        for key, value in hashes.items():
            yield f"{key}:"
            yield from (map(str, value))


class AptMirror(RepoDownload):
    """Wrapper around the `apt-mirror` tool."""

    SOURCES_PATH = Path("/etc/apt/sources.list.d")
    MIRROR_CONFIG_PATH = Path("/etc/apt/mirror.list")

    _known_repos: Optional[Set[str]]

    def __init__(self):
        super().__init__(SubprocessWrapper("apt-mirror"))
        self._known_repos = None

    def clone_repos(
        self,
        targets: Sequence[str],
        output: Path,
        collapse: bool,
        architectures: Iterable[str],
        *_args: Any,
        **_kwargs: Any,
    ):
        entries: List[str] = []

        for target in targets:
            if target in self.known_repos:
                entries.extend(self._get_entries_from_known_repo(target))
            else:
                # Working with a url with possible codename and component
                entries.append(target)

        entries = [
            qualified_entry
            for entry in entries
            for qualified_entry in self._add_architecture_to_entry(entry, architectures)
        ]

        with TemporaryDirectory() as tempdir, self._backup_mirror_list(Path(tempdir)):
            self._write_mirror_config(
                self.MIRROR_CONFIG_PATH, output, tempdir, collapse, entries
            )
            _moduleLogger.info("Starting repo clone")
            self.program.execute()

            if collapse:
                self._collapse_path(Path(tempdir, "mirror"), output)

    @property
    def known_repos(self) -> Set[str]:
        """Returns the set of known repos on the system."""
        if self._known_repos is None:
            self._known_repos = set(
                (path.stem for path in self.SOURCES_PATH.glob("*.list"))
            )
        return self._known_repos

    @staticmethod
    def _add_architecture_to_entry(
        entry: str, architectures: Iterable[str]
    ) -> Iterable[str]:
        deb, rest = entry.split(maxsplit=1)
        if deb is None:
            raise ValueError(f"'{entry}' is not a valid entry.")

        if deb == "deb":
            for architecture in architectures:
                yield f"{deb}-{architecture} {rest}"
        else:
            yield entry

    @classmethod
    @contextlib.contextmanager
    def _backup_mirror_list(cls, temp: Path) -> Generator[None, None, None]:
        mirror_config_backup = cls.MIRROR_CONFIG_PATH.exists() and temp.joinpath(
            "mirror.list.back"
        )
        if mirror_config_backup:
            # Save
            cls.MIRROR_CONFIG_PATH.rename(mirror_config_backup)

        yield

        if mirror_config_backup:
            # Restore
            mirror_config_backup.rename(cls.MIRROR_CONFIG_PATH)

    @staticmethod
    def _collapse_path(path: Path, output: Path):
        """Delete empty directories between specified path and the first
        directory with more than one child."""
        _moduleLogger.debug("Collapsing path")

        cur_path = path
        while True:
            items = list(itertools.islice(cur_path.iterdir(), 2))
            if len(items) != 1:
                break
            cur_path = cur_path.joinpath(items[0])

        for item in cur_path.iterdir():
            target = output.joinpath(item.name)
            shutil.move(str(item), target)

    @classmethod
    def _get_entries_from_known_repo(cls, repo_name: str) -> Iterable[str]:
        with cls.SOURCES_PATH.joinpath(f"{repo_name}.list").open(
            "r", encoding="utf-8"
        ) as target_file:
            yield from (
                line
                for line in map(str.strip, target_file.readlines())
                if line and not line.startswith("#")
            )

    @staticmethod
    def _write_mirror_config(
        path: Path, output: Path, temp: str, collapse: bool, entries: Sequence[str]
    ):
        with path.open("w") as config_file, contextlib.redirect_stdout(config_file):
            print(f"set base_path {temp}")
            if not collapse:
                print(f"set mirror_path {str(output)}")
            print("set nthreads 20")
            print("set _tilde 0")
            for entry in entries:
                print(entry)


class StoreDictAction(argparse.Action):  # pylint: disable=too-few-public-methods
    """Argument action to accumulate key-value pairs into a dictionary."""

    def __call__(
        self,
        _parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Optional[Union[str, Sequence[Any]]],
        _option_string: Optional[str] = None,
    ):
        assert values is not None
        try:
            key, value = re.split(r"[:=]", str(values), maxsplit=1)
        except ValueError:
            key, value = (values, None)

        metadata = getattr(namespace, self.dest, {})

        if metadata is None:
            metadata = {}
        metadata[key] = value

        setattr(namespace, self.dest, metadata)


def get_parser() -> argparse.ArgumentParser:
    """Get the argument parser for this script."""
    short_description, _, long_description = __doc__.split("\n", 2)  # type: ignore
    examples = """

Examples:

    Create a repository for the GPIB driver and its dependencies
        %(prog)s packages ni-488.2

    Create a repository containing both the NI-VISA and NI-Serial drivers and their
    dependencies in a directory named "my_repo" while ignoring suggested and
    recommended packages
        %(prog)s --output my_repo packages --exclude-suggests --exclude-recommends ni-visa ni-serial

    Clone the ni-software-2022 repository with max verbosity logging
        %(prog)s -vv repo --collapse ni-software-2022-focal

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
        "--exclude-recommends",
        action="store_true",
        help="exclude recommended dependencies",
    )
    packages_parser.add_argument(
        "--exclude-suggests",
        action="store_true",
        help="exclude suggested dependencies",
    )
    packages_parser.add_argument(
        "--exclude-system",
        action="store_true",
        help="exclude installed and system packages",
    )
    packages_parser.add_argument(
        "-s",
        "--sources-list",
        dest="sources_list",
        metavar="PATH",
        help="use the specified sources.list file instead of the system file",
    )

    metadata_group = packages_parser.add_argument_group(
        "metadata arguments",
        description="arguments to specify metadata to add to the created repository",
    )
    metadata_group.add_argument(
        "-A",
        "--arch",
        action="append",
        default=[],
        metavar="ARCH",
        dest="architectures",
        help='architectures for this repository (can be repeated) (default: "amd64", "i386")',
    )
    metadata_group.add_argument(
        "-C",
        "--component",
        default="main",
        help='component for this repository in the SUITE (default: "main")',
    )
    metadata_group.add_argument(
        "-S",
        "--suite",
        default="stable",
        help='suite for this repository in the OUTPUT distribution (default: "stable")',
    )
    metadata_group.add_argument(
        "-M",
        "--metadata",
        action=StoreDictAction,
        default={},
        metavar="KEY:VALUE",
        help="add additional metadata to the created repository (can be repeated)",
    )

    sign_group = packages_parser.add_argument_group("signing arguments")
    sign_group.add_argument("-K", "--key-id", help="id of GPG key to use for signing")
    sign_group.add_argument(
        "-p",
        "--passphrase",
        const="-",
        nargs="?",
        help="passphrase for GPG key (can be file path) (pass '-' to read from stdin)",
    )
    sign_group.add_argument(
        "--no-passphrase",
        action="store_true",
        help="do not prompt for passphrase",
    )
    sign_group.add_argument(
        "--no-sign",
        action="store_true",
        help="do not sign the resulting repository",
    )

    packages_parser.set_defaults(operation=PackagesDownload)

    repo_parser = subparsers.add_parser("repo", help="clone existing repositories")
    repo_parser.add_argument(
        "targets",
        metavar="TARGET",
        nargs="+",
        help="target repositories to clone (specify by name or source entry)",
    )
    repo_parser.add_argument(
        "-A",
        "--arch",
        action="append",
        default=[],
        metavar="ARCH",
        dest="architectures",
        help='architectures for this repository (can be repeated) (default: "amd64", "i386")',
    )
    repo_parser.add_argument(
        "--collapse",
        action="store_true",
        help="collapse empty directories in output",
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
    verbosity = min(max(verbosity, 0), 4)
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
    except Exception as err:  # pylint: disable=broad-except,redefined-outer-name
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
