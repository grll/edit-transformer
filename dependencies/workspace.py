import os
from os.path import join
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