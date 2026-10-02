from toolbox.logger import get_logger
import pandas as pd

logger = get_logger(__name__)

class AuditTrail:
    def __init__(self):
        self.trail = []

    def log(self, step, column, details):
        """
        Description: Adds a log to an established audit trail

        Input:
        step - specified operation
        column - column used for the current step
        details - details about the operation
        """
        record = {
            "timestamp": pd.Timestamp.now(),
            "step": step,
            "column": column,
            "details": details
        }
        self.trail.append(record)
        logger.debug(f"Audit logged — {step} on '{column}': {details}")

    def summary(self):
        """
        Generates a report about the current Audit Trail
        """
        if not self.trail:
            logger.info("Audit trail is empty — no transformations applied yet.")
            return

        print("----------- Audit Trail -------------------")
        for i, record in enumerate(self.trail, 1):
            print(f"{i}. [{record['timestamp']}] {record['step']} | {record['column']} | {record['details']}")
        print(f"Total steps applied: {len(self.trail)}")

    def export(self, output="excel", path=None):
        """
        Exports Audit Trail contents to HTML or Excel

        Input:
        output - (Default, Excel) 'html' or 'excel'
        path - output path to store Audit Trail contents in chosen format
        """
        if path is None:
            path = f"audit_trail.{output}"
            logger.warning(f"No path provided, saving to {path}")

        df = self.to_dataframe()

        if output == "excel":
            df.to_excel(path, index=False)
            logger.info(f"Audit trail exported to {path}")
        elif output == "html":
            df.to_html(path, index=False)
            logger.info(f"Audit trail exported to {path}")
        else:
            logger.error(f"Unsupported output format: {output}")
            raise ValueError(f"Unsupported output format: {output}")

    def to_dataframe(self):
        """
        Converts Audit Trail to Dataframe
        """
        return pd.DataFrame(self.trail)

    def clear(self):
        """
        Clears Audit Trail
        """
        self.trail = []
        logger.debug("Audit trail cleared")

    def __len__(self):
        """
        Returns Length of Audit Trail
        """
        return len(self.trail)