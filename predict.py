from sklearn.model_selection import train_test_split
from sklearn import datasets
import json
import boto3

global app_name
global region
app_name = 'model-application'
region = 'ap-south-1'

def check_status(app_name):
    sage_client = boto3.client('sagemaker', region_name=region)
    endpoint_description = sage_client.describe_endpoint(EndpointName=app_name)
    endpoint_status = endpoint_description["EndpointStatus"]
    return endpoint_status

def query_endpoint(app_name, input_json):
    client = boto3.session.Session().client("sagemaker-runtime", region)

    response = client.invoke_endpoint(
        EndpointName=app_name,
        Body=input_json,
        ContentType='application/json',
    )
    preds = response['Body'].read().decode("ascii")
    preds = json.loads(preds)
    print("Received response: {}".format(preds))
    return preds

## check endpoint status
print("Application status is: {}".format(check_status(app_name)))

# Prepare data to give for predictions
iris = datasets.load_iris()
x = iris.data[:, 2:]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7)

## create test data and make inference from enpoint
query_input = {
    "dataframe_split": {
        "columns": ["petal length (cm)", "petal width (cm)"],
        "data": X_test[0:3].tolist(),
    }
}

import json

query_input = json.dumps(query_input)


print("Query input: {}".format(query_input))
prediction = query_endpoint(app_name=app_name, input_json=bytes(query_input, 'utf-8'))  