from toolbox.logger import get_logger
import pandas as pd

logger = get_logger(__name__)

class AuditTrail:
    def __init__(self):
        self.trail = []

    def log(self, step, column, details):
        record = {
            "timestamp": pd.Timestamp.now(),
            "step": step,
            "column": column,
            "details": details
        }
        self.trail.append(record)
        logger.debug(f"Audit logged — {step} on '{column}': {details}")

    def summary(self):
        if not self.trail:
            logger.info("Audit trail is empty — no transformations applied yet.")
            return

        print("----------- Audit Trail -------------------")
        for i, record in enumerate(self.trail, 1):
            print(f"{i}. [{record['timestamp']}] {record['step']} | {record['column']} | {record['details']}")
        print(f"Total steps applied: {len(self.trail)}")

    def export(self, output="excel", path=None):
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
        return pd.DataFrame(self.trail)

    def clear(self):
        self.trail = []
        logger.debug("Audit trail cleared")

    def __len__(self):
        return len(self.trail)