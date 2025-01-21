import ctypes
import ctypes.util

try:
    objc = ctypes.cdll.LoadLibrary('libobjc.dylib')
except OSError:
    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library('objc'))

void_p = ctypes.c_void_p

objc.objc_getClass.restype = void_p
objc.sel_registerName.restype = void_p

# See https://docs.python.org/3/library/ctypes.html#function-prototypes for arguments description
MSGPROTOTYPE = ctypes.CFUNCTYPE(void_p, void_p, void_p, void_p)
msg = MSGPROTOTYPE(('objc_msgSend', objc), ((1, '', None), (1, '', None), (1, '', None)))


def _utf8(s):
    if not isinstance(s, bytes):
        s = s.encode('utf8')
    return s


def n(name):
    return objc.sel_registerName(_utf8(name))


def C(classname):
    return objc.objc_getClass(_utf8(classname))


def theme():
    ns_autorelease_pool = objc.objc_getClass('NSAutoreleasePool')
    pool = msg(ns_autorelease_pool, n('alloc'))
    pool = msg(pool, n('init'))

    ns_user_defaults = C('NSUserDefaults')
    std_user_def = msg(ns_user_defaults, n('standardUserDefaults'))

    ns_string = C('NSString')

    key = msg(ns_string, n("stringWithUTF8String:"), _utf8('AppleInterfaceStyle'))
    appearance_ns = msg(std_user_def, n('stringForKey:'), void_p(key))
    appearance_c = msg(appearance_ns, n('UTF8String'))

    if appearance_c is not None:
        out = ctypes.string_at(appearance_c)
    else:
        out = None

    msg(pool, n('release'))

    if out is not None:
        return out.decode('utf-8')
    else:
        return 'Light'


def isDark():
    return theme() == 'Dark'


def isLight():
    return theme() == 'Light'
