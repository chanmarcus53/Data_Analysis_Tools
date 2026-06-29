from toolbox.logger import get_logger
from toolbox.cleaning.audit import AuditTrail
from toolbox.cleaning.missing import handle_missing
from toolbox.cleaning.outliers import handle_outliers

logger = get_logger(__name__)

VALID_STEPS = ["handle_missing", "handle_outliers"]

class Pipeline:
    def __init__(self, steps=None):
        self.steps = []
        self.audit = AuditTrail()
        for step in (steps or []):
            self.add_step(step)

    def run(self, df):
        self.audit.clear()
        logger.info(f"Pipeline starting — {len(self.steps)} steps to apply")

        for i, step in enumerate(self.steps, 1):
            step_name = step.get("step")
            column = step.get("column")
            logger.info(f"Step {i}/{len(self.steps)}: {step_name} on '{column}'")

            kwargs = {k: v for k, v in step.items() if k != "step"}

            if step_name == "handle_missing":
                df = handle_missing(df, audit=self.audit, **kwargs)
            elif step_name == "handle_outliers":
                df = handle_outliers(df, audit=self.audit, **kwargs)

        logger.info("Pipeline complete")
        return df

    def add_step(self, step):
        if "step" not in step:
            raise ValueError("Each step must have a 'step' key")
        if "column" not in step:
            raise ValueError("Each step must have a 'column' key")
        if step["step"] not in VALID_STEPS:
            raise ValueError(f"Unknown step: '{step['step']}'. Choose from: {VALID_STEPS}")
        self.steps.append(step)
        logger.debug(f"Step added: {step['step']} on column '{step['column']}'")

    def summary(self):
        print("----------- Pipeline Steps -------------------")
        if not self.steps:
            print("No steps defined.")
        else:
            for i, step in enumerate(self.steps, 1):
                details = {k: v for k, v in step.items() if k != "step"}
                print(f"{i}. {step['step']} — {details}")

        print("\n----------- Audit Trail ---------------------")
        if len(self.audit) == 0:
            print("Pipeline has not been run yet.")
        else:
            self.audit.summary()