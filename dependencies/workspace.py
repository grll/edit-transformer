import os
from os.path import join
from collections import Mapping, OrderedDict
import mimetypes
mimetypes.init()  # see Workspace


def makedirs(directory):
    """If directory does not exist, make it.

    Args:
        directory (str): a path to a directory. Cannot be the empty path.
    """
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)


class Workspace(object):
    """Manage paths underneath a top-level root directory.

    Paths are registered with this Workspace. An IOError is thrown if the path has already been registered before.
    """

    def __init__(self, root):
        """Create a Workspace.

        Args:
            root (str): absolute path of the top-level directory.
        """
        self._root = root
        makedirs(root)
        self._paths = set()

    @property
    def root(self):
        return self._root

    def __getattr__(self, relative_path):
        """Allow access to unregistered files and directories

        attr_path is used to consume and build up absolute paths
        e.g. workspace.a.b.c -> "{root}/a".b.c -> "{root}/a/b".c -> ...
        """

        class attr_path(str):
            def __new__(cls, path):
                cls.path = path
                return super(attr_path, cls).__new__(cls, path)

            def __getattr__(cls, next):
                if ('.' + next) in mimetypes.types_map:
                    return attr_path(cls.path + ('.' + next))  # file access
                return attr_path(join(cls.path, next))  # dir access

        return attr_path(self._root).__getattr__(relative_path)

    def hasattr(self, attr):
        """Use instead of hasattr for this class. It turns out hasattr invokes
        __getattr__, so our overridden __getattr__ hack will always
        evaluate to true, which is a problem (see above).
        """
        return attr in self.__dict__

    def _add(self, name, relative_path):
        """Register a path.

        Args:
            name (str): short name to reference the path
            relative_path (str): a relative path, relative to the workspace root.

        Returns:
            self
        """
        full_path = join(self._root, relative_path)
        if self.hasattr(name):
            raise IOError('Name already registered: {}'.format(name))
        if full_path in self._paths:
            raise IOError('Path already registered: {}'.format(relative_path))
        setattr(self, name, full_path)

    def add_dir(self, name, relative_path):
        self._add(name, relative_path)
        makedirs(getattr(self, name))

    def add_file(self, name, relative_path):
        self._add(name, relative_path)


def sub_dirs(root_dir):
    """Return a list of all sub-directory paths.

    Example:
        >> root_dir = '/Users/Kelvin/data'
        >> sub_dirs(root_dir)
        ['/Users/Kelvin/data/a', '/Users/Kelvin/data/b']
    """
    dir_paths = []
    for path in os.listdir(root_dir):
        full_path = join(root_dir, path)
        if os.path.isdir(full_path):
            dir_paths.append(full_path)
    return dir_paths


class IntegerDirectories(Mapping):
    """Keep track of directories with names of the form "{integer}_{something}" or just "{integer}"."""
    def __init__(self, root_dir):
        self.root_dir = root_dir
        makedirs(root_dir)

    @property
    def _ints_to_paths(self):
        ints_to_paths = {}
        for p in sub_dirs(self.root_dir):
            name = os.path.basename(p)
            try:
                i = int(name.split('_')[0])
                if i in ints_to_paths:
                    raise IOError("Multiple directories with the same integer prefix: {} and {}".format(
                        ints_to_paths[i], p))
                ints_to_paths[i] = p
            except ValueError:
                # the first element was not an integer
                pass

        # put into an ordered dict
        ordered = OrderedDict()
        for i in sorted(ints_to_paths):
            ordered[i] = ints_to_paths[i]
        return ordered

    def __len__(self):
        return len(self._ints_to_paths)

    @property
    def largest_int(self):
        """Largest int among the integer directories."""
        if len(self._ints_to_paths) == 0:
            return None
        return max(self._ints_to_paths)

    def new_dir(self, name=None):
        """Create a new directory and return its path."""
        if self.largest_int is None:
            idx = 0
        else:
            idx = self.largest_int + 1

        path = join(self.root_dir, str(idx))

        if name:
            path = '{}_{}'.format(path, name)  # add name as suffix

        makedirs(path)
        return path

    def __getitem__(self, i):
        """Get the path to directory i.

        Raises:
            KeyError, if directory does not exist.
        """
        if i not in self._ints_to_paths:
            raise KeyError("Directory #{} not found".format(i))
        return self._ints_to_paths[i]

    def __iter__(self):
        return iter(self._ints_to_paths)
