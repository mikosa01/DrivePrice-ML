from sklearn.model_selection import train_test_split
from classification.class_model.data_management.data_management import load_dataset, save_pipeline
from classification.class_model.config import config
from classification.class_model.pipeline import pipeline
from tracking import model_tracking


def running():
    data = load_dataset(file_name=config.DATA_NAME) 
    # Split the dataset into training and testing sets based on the feature and target columns.
    # The split is done using the specified test size and random state from the config.
    x_train, x_test, y_train, y_test = train_test_split(
        data[config.FEATS_COLUMNS], 
        data[config.TARGET_COLUMNS], 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_SIZE
    )
    # Fit the pipeline (which includes all preprocessing steps and the model) on the training data.
    pipeline.fit(x_train, y_train)
    # Save the fitted pipeline to the specified location.
    save_pipeline(pipeline_persit=pipeline)

if __name__ == "__main__":
    # Run the main function to execute the pipeline workflow.
    running()