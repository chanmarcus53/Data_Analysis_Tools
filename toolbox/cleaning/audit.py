from toolbox.logger import get_logger

logger = get_logger(__name__)

class AuditTrail:
    def __init__(self):
        # TODO: what data structure would you use to store the steps?
        # hint: you'll be appending records over time
        raise NotImplementedError

    def log(self, step, column, details):
        # TODO: record a transformation
        # think about what information is useful to capture beyond
        # just step, column and details
        # hint: what else would you want to know when reviewing the trail?
        raise NotImplementedError

    def summary(self):
        # TODO: print a readable summary of all steps applied
        # hint: follow the same pattern as _print_console in report.py
        raise NotImplementedError

    def export(self, output="excel", path=None):
        # TODO: save the audit trail to Excel or HTML
        # hint: you already solved this problem in report.py
        raise NotImplementedError

    def clear(self):
        # TODO: reset the trail
        raise NotImplementedError