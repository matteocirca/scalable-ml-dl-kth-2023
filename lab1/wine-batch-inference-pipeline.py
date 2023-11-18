import os
import modal
    
LOCAL=False

if LOCAL == False:
   stub = modal.Stub("wine_batch")
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","scikit-learn==1.1.1","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("iris"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=2)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")
    
    feature_view = fs.get_feature_view(name="wine", version=4)
    batch_data = feature_view.get_batch_data()
    
    y_pred = model.predict(batch_data)
    #print(y_pred)
    offset = 3
    wine = y_pred[y_pred.size-offset]
    print("Wine predicted: " + wine)
   
    wine_fg = fs.get_feature_group(name="wine", version=4)
    df = wine_fg.read() 
    #print(df)
    label = df.iloc[-offset]["quality"]
    print("Wine actual: " + str(label))
    
    monitor_fg = fs.get_or_create_feature_group(name="wine_predictions",
                                                version=2,
                                                primary_key=["datetime"],
                                                description="Wine Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [wine],
        'label': [label],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])


    df_recent = history_df.tail(4)
    dfi.export(df_recent, './df_wine_recent.png', table_conversion = 'matplotlib')
    dataset_api = project.get_dataset_api()
    dataset_api.upload("./df_wine_recent.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our iris_predictions feature group has examples of all good, bad
    print("Number of different flower predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 2:
        results = confusion_matrix(labels, predictions)
    
        df_cm = pd.DataFrame(results, ['True Good', 'True Bad'],
                             ['Pred Good', 'Pred Bad'])
    
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix_wine.png")
        dataset_api.upload("./confusion_matrix_wine.png", "Resources/images", overwrite=True)
    else:
        print("You need 2 wines to create the confusion matrix, one of good quality and the other one of bad quality.")
        print("Run the batch inference pipeline more times until you get 2 different wine predictions") 


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("wine_batch")
        with stub.run():
            f()

