import platform
import sys


def macos_supported_version():
    sys_ver = platform.mac_ver()[0]  # typically 10.14.2 or 12.3
    major = int(sys_ver.split('.')[0])
    if major < 10:
        return False
    elif major >= 11:
        return True
    else:
        minor = int(sys_ver.split('.')[1])
        if minor < 14:
            return False
        else:
            return True


if sys.platform == "darwin":
    if macos_supported_version():
        from ._mac_detect import *
    else:
        from ._dummy import *
elif sys.platform == "win32" and platform.release().isdigit() and int(platform.release()) >= 10:
    # Checks if running Windows 10 version 10.0.14393 (Anniversary Update) OR HIGHER. The get Windows version method
    # returns a tuple. The third item is the build number that we can use to check if the user has a new enough
    # version of Windows.
    win_ver = int(platform.version().split('.')[2])
    if win_ver >= 14393:
        from ._windows_detect import *
    else:
        from ._dummy import *
elif sys.platform == "linux":
    from ._linux_detect import *
else:
    from ._dummy import *

del sys, platform
