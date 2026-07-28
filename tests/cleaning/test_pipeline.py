from toolbox.cleaning.pipeline import Pipeline
import pandas as pd
import pytest

class TestInit:
    def test_pipline_init(self):
        pipeline = Pipeline()
        assert pipeline.steps == []

    def test_pipeline_init_with_steps(self, sample_pipeline_steps):
        pipeline = Pipeline(steps=sample_pipeline_steps)
        assert len(pipeline.steps) == 2

    def test_pipeline_audit_trail_init(self):
        pipeline = Pipeline()
        assert pipeline.audit.trail == []

    def test_pipeline_add_step_invalid_step(self):
        pipeline = Pipeline()
        with pytest.raises(ValueError):
            pipeline.add_step({"step": "invalid_step", "column": "age"})
        assert len(pipeline.steps) == 0

    def test_pipeline_incalid_step_at_init(self):
        with pytest.raises(ValueError):
            Pipeline(steps=[{"step": "invalid_step", "column": "age"}])


class TestAddStep:
    def test_add_step_valid(self):
        pipeline = Pipeline()
        step = {"step": "handle_missing", "column": "age", "strategy": "median"}
        pipeline.add_step(step)
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0] == step

    def test_missing_step_key(self):
        pipeline = Pipeline()
        with pytest.raises(ValueError):
            pipeline.add_step({"column": "age", "strategy": "median"})
        assert len(pipeline.steps) == 0

    def test_missing_column_key(self):
        pipeline = Pipeline()
        with pytest.raises(ValueError):
            pipeline.add_step({"step": "handle_missing", "strategy": "median"})
        assert len(pipeline.steps) == 0

    def test_unknown_step_name(self):
        pipeline = Pipeline()
        with pytest.raises(ValueError):
            pipeline.add_step({"step": "unknown_step", "column": "age"})
        assert len(pipeline.steps) == 0


class TestRun:
    def test_returns_dataframe(self, outlier_df, sample_pipeline_steps):
        pipeline = Pipeline(steps=sample_pipeline_steps)
        result_df = pipeline.run(outlier_df)
        assert isinstance(result_df, pd.DataFrame)

    def test_run_operation(self, outlier_df, sample_pipeline_steps):
        pipeline = Pipeline(steps=sample_pipeline_steps)
        result_df = pipeline.run(outlier_df)
        assert result_df is not None
        assert result_df.iloc[3]["age"] == 100 
        assert result_df.iloc[3]["age_outlier"] == True # Outlier should be flagged, not removed

    def test_audit_trail_after_run(self, null_df, sample_pipeline_steps):
        pipeline = Pipeline(steps=sample_pipeline_steps)
        pipeline.run(null_df)
        assert len(pipeline.audit.trail) > 0
        assert pipeline.audit.trail[0]["step"] == "handle_missing"
        assert pipeline.audit.trail[1]["step"] == "handle_outliers"

    def test_audit_trail_reset(self, sample_pipeline_steps, null_df):
        pipeline = Pipeline(steps=sample_pipeline_steps)
        pipeline.run(null_df)
        first_run_count = len(pipeline.audit.trail) > 0
        pipeline.run(null_df)
        assert len(pipeline.audit.trail) == first_run_count

    def test_run_applies_missing_strategy(self, null_df, sample_pipeline_steps):
        pipeline = Pipeline(steps=sample_pipeline_steps)
        result_df = pipeline.run(null_df)
        # first step is handle_missing with median strategy
        assert result_df["age"].isnull().sum() == 0

    def test_run_does_not_modify_original(self, null_df, sample_pipeline_steps):
        original = null_df.copy()
        pipeline = Pipeline(steps=sample_pipeline_steps)
        pipeline.run(null_df)
        pd.testing.assert_frame_equal(null_df, original)

    def test_run_empty_pipeline(self, null_df):
        pipeline = Pipeline()
        result_df = pipeline.run(null_df)
        pd.testing.assert_frame_equal(result_df, null_df)

        
class TestSummary:
    def test_summary_after_run(self, null_df, sample_pipeline_steps, capsys):
        pipeline = Pipeline(steps=sample_pipeline_steps)
        pipeline.run(null_df)
        pipeline.summary()
        captured = capsys.readouterr()
        assert "handle_missing" in captured.out
        assert "handle_outliers" in captured.out

    def test_summary_empty_pipeline(self, capsys):
        pipeline = Pipeline()
        pipeline.summary()
        captured = capsys.readouterr()
        assert "No steps defined" in captured.out

    def test_summary_not_run_yet(self, sample_pipeline_steps, capsys):
        pipeline = Pipeline(steps=sample_pipeline_steps)
        pipeline.summary()
        captured = capsys.readouterr()
        assert "not been run yet" in captured.out



