import rich

_log_styles = {
    "MonoGS": "bold green",
    "OnlineGS": "bold green",
    "GUI": "bold magenta",
    "Eval": "bold red",
    "GauSS-MI": "bold blue",
    "ActiveManage": "bold cyan",
}

def get_style(tag):
    if tag in _log_styles.keys():
        return _log_styles[tag]
    return "bold blue"

def Log(*args, tag="GauSS-MI"):
    style = get_style(tag)
    rich.print(f"[{style}]{tag}:[/{style}]", *args)
