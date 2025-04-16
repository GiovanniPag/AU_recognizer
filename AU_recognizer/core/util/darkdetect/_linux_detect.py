import subprocess


def theme():
    try:
        # Using the freedesktop specifications for checking dark mode
        out = subprocess.run(
            ['gsettings', 'get', 'org.gnome.desktop.interface', 'color-scheme'],
            capture_output=True)
        stdout = out.stdout.decode()
        # If not found then trying older gtk-theme method
        if len(stdout) < 1:
            out = subprocess.run(
                ['gsettings', 'get', 'org.gnome.desktop.interface', 'gtk-theme'],
                capture_output=True)
            stdout = out.stdout.decode()
    except Exception as err:
        print(err)
        return 'Light'
    # we have a string, now remove start and end quote
    theme_ = stdout.lower().strip()[1:-1]
    if '-dark' in theme_.lower():
        return 'Dark'
    else:
        return 'Light'


def isDark():
    return theme() == 'Dark'


def isLight():
    return theme() == 'Light'
