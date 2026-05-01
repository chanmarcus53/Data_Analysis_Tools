def report(profile_result, output=None):
    """
    output=None prints to console.
    output="html" or output="excel" saves a file.
    """
    if output is None:
        _print_console(profile_result)
    elif output == "html":
        _export_html(profile_result)
    elif output == "excel":
        _export_excel(profile_result)
    else:
        raise ValueError(f"Unsupported output format: {output}")


def _print_console(profile_result):
    # TODO: print a readable summary — think about what a person
    # actually needs to see first when they load a new dataset
    raise NotImplementedError

def _export_html(profile_result):
    # TODO: generate a simple HTML file from the profile dict
    # hint: look into Jinja2 templating, or even just f-strings for a start
    raise NotImplementedError

def _export_excel(profile_result):
    # TODO: write profile to Excel with one sheet per section
    # hint: look into pd.ExcelWriter and openpyxl
    raise NotImplementedError