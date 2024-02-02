import boto3
from botocore.exceptions import NoCredentialsError, ClientError


def configure_aws_session(aws_access_key_id, aws_secret_access_key, aws_session_token=None, region_name='us-east-1'):
    """
    Configure AWS session with the provided credentials.

    :param aws_access_key_id: AWS access key ID
    :param aws_secret_access_key: AWS secret access key
    :param aws_session_token: AWS session token (optional)
    :param region_name: AWS region name
    :return: None
    """
    boto3.setup_default_session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        region_name=region_name
    )


def upload_to_s3(file_name, bucket, object_name=None):
    """
    Upload a file to an S3 bucket.

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    if object_name is None:
        object_name = file_name

    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(file_name, bucket, object_name)
    except NoCredentialsError:
        print("Credentials not available")
        return False
    return True

def download_from_s3(bucket, object_name, file_name=None):
    """
    Download a file from an S3 bucket.

    :param bucket: Bucket to download from
    :param object_name: S3 object name
    :param file_name: File name to save the downloaded content. 
                      If not specified then object_name is used
    :return: True if file was downloaded, else False
    """
    if file_name is None:
        file_name = object_name

    s3_client = boto3.client('s3')
    try:
        s3_client.download_file(bucket, object_name, file_name)
    except NoCredentialsError:
        print("Credentials not available")
        return False
    return True


def validate_aws_credentials(bucket):
    """
    Validate AWS IAM credentials by attempting to list objects in the specified bucket.

    :param bucket: S3 bucket name to test the connection.
    :return: True if credentials are valid and bucket is accessible, else False.
    """
    s3_client = boto3.client('s3')
    try:
        s3_client.list_objects_v2(Bucket=bucket, MaxKeys=1)
        return True
    except NoCredentialsError:
        print("No credentials provided.")
    except ClientError as e:
        if e.response['Error']['Code'] == '403':
            print("Access to the bucket denied.")
        elif e.response['Error']['Code'] == '404':
            print("The bucket does not exist.")
        else:
            print("Error occurred: " + str(e))
    return False


def aws_credentials_configured():
    """Check if AWS credentials are configured."""
    try:
        boto3.client('sts').get_caller_identity()
        return True
    except NoCredentialsError:
        return False
    
def get_files_list_from_s3(bucket_name):
    """
    Retrieves the list of files from the specified S3 bucket.
    
    :param bucket_name: Name of the S3 bucket
    :return: List of file names in the bucket
    """
    s3_client = boto3.client('s3')
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        files = []
        if 'Contents' in response:
            files = [obj['Key'] for obj in response['Contents']]
        return files
    except Exception as e:
        print(f"Error getting files from S3: {e}")
        return []
    

def list_s3_buckets():

    s3_client = boto3.client('s3')
    response = s3_client.list_buckets()
    return [bucket['Name'] for bucket in response['Buckets']]

def list_files_in_bucket(bucket_name):

    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    return [obj['Key'] for obj in response.get('Contents', [])]

# def download_file_from_s3(bucket_name, file_key, local_path):

#     s3_client = boto3.client('s3')
#     s3_client.download_file(bucket_name, file_key, local_path)