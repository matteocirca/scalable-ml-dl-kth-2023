import os
import modal

LOCAL=False

if LOCAL == False:
   stub = modal.Stub("wine_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks"])

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("wine"))
   def f():
       g()


def get_random_wine(df, model):
    """
    Returns a DataFrame containing two synthetic wine samples, one white and one red
    """
    import pandas as pd
    import random
    import numpy as np

    feat = {'white': [], 'red': []}

    # white wine is 0, red wine is 1
    white_df = df[df['type'] == 0]
    red_df = df[df['type'] == 1]

    for x in df.columns[:-1]:
        std1 = white_df[x].std()
        mean1 = white_df[x].mean()

        std2 = red_df[x].std()
        mean2 = red_df[x].mean()

        feat['white'].append(np.random.normal(mean1, std1, 1)[0])
        feat['red'].append(np.random.normal(mean2, std2, 1)[0])
        
    df = pd.DataFrame(columns=df.columns[:-1], data=[feat['white'], feat['red']])
    df['quality'] = model.predict(df)

    print(df)

    return df


def g():
    import pandas as pd
    import hopsworks
    import joblib

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")

    wine_fg = fs.get_feature_group(name="wine", version=3)
    df = wine_fg.read()

    wine_df = get_random_wine(df, model)

    wine_fg.insert(wine_df)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("iris_daily")
        with stub.run():
            f()
