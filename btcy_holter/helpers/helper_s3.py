import os
import json
import boto3
import botocore

from botocore.config import Config
from botocore.exceptions import (
    NoCredentialsError,
    PartialCredentialsError,
    ClientError,
    ProfileNotFound
)

from typing import Final, Any, List
import btcy_holter.stream as st
import btcy_holter.config as cf


class HelperS3(
        object
):
    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html
    
    NAME:                   Final[str] = 's3'
    
    RETRY:                  Final[int] = 5
    MAX_POOL_CONNECTIONS:   Final[int] = 25
    
    def __init__(
            self
    ) -> None:
        try:
            self.bucket_name:       Final[str] = cf.AWS_BUCKET
            self.use_irsa:          Final[bool] = cf.USE_IRSA

            self._client_s3 = self._create_client_session()

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

    def _create_client_session(
            self
    ) -> Any:
        client = None
        try:
            st.LOGGING_SESSION.info(f'[{self.NAME}] - [use IRSA: {cf.USE_IRSA}] - aws profile: {cf.AWS_PROFILE}')
            if cf.DEBUG_MODE and self.check_aws_profile_exists(cf.AWS_PROFILE):
                client_config = botocore.config.Config(
                        max_pool_connections=self.MAX_POOL_CONNECTIONS
                )
                
                session = boto3.session.Session(
                        profile_name=cf.AWS_PROFILE
                )
                client = session.client(
                        service_name=self.NAME,
                        region_name=cf.AWS_REGION,
                        config=client_config
                )

            elif self.use_irsa:
                session = boto3.session.Session(
                        profile_name=cf.AWS_PROFILE
                )
                client = session.client(
                        service_name=self.NAME,
                        region_name=cf.AWS_REGION
                )
                
            else:
                client = boto3.client(
                        service_name=self.NAME,
                        region_name=cf.AWS_REGION
                )

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return client

    def upload_file(
            self,
            local_file:     str,
            s3_file:        str,
            bucket_name:    str = None,
            debug_mode:     bool = cf.DEBUG_MODE,
            show_log:       bool = True,
    ) -> bool:
        """
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/upload_file.html
        """
        is_success = False
        try:
            if debug_mode:
                return True

            if not os.path.exists(local_file):
                show_log and st.LOGGING_SESSION.warning(
                        f'--- Upload {s3_file} is failed, b/c the local file does not exist!'
                )
                return is_success
            
            if bucket_name is None:
                bucket_name = self.bucket_name
                
            self._client_s3.upload_file(
                    Filename=local_file,
                    Key=s3_file,
                    Bucket=bucket_name
            )
            is_success = True
            show_log and st.LOGGING_SESSION.info(f'--- Upload {s3_file} is success!')

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return is_success

    def download_file(
            self,
            local_file:     str,
            s3_file:        str,
            bucket_name:    str = None,
            show_log:       bool = True,
    ) -> bool:
        """
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/download_file.html
        """
        is_success = False
        try:
            if os.path.exists(local_file) and os.path.isfile(local_file):
                return True
            
            if bucket_name is None:
                bucket_name = self.bucket_name
                
            is_success = self.check_prefix_key_exist(
                    bucket_name=bucket_name,
                    s3_file=s3_file
            )
            if not is_success:
                show_log and st.LOGGING_SESSION.warning(
                        f'--- Download {s3_file} is failed, b/c the local file does not exist!'
                )
                return is_success
                
            self._client_s3.download_file(
                    Bucket=bucket_name,
                    Key=s3_file,
                    Filename=local_file
            )
            is_success = True
            
            show_log and st.LOGGING_SESSION.info(f'--- Download {s3_file} is success!')

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return is_success

    def get_s3_file(
            self,
            object_name: str,
            byte_range:  list = None,
    ) -> Any:
        body = None
        try:
            is_existed = self.check_prefix_key_exist(
                    s3_file=object_name
            )
            if is_existed:
                res = self._client_s3.get_object(
                    Bucket=self.bucket_name,
                    Key=object_name,
                    Range=byte_range
                )
                body = res['Body'].read()
            else:
                st.LOGGING_SESSION.info(f'Could not download file {object_name}, '
                                        f'because file does not exist')

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return body

    def get_files_by_prefix(
            self,
            s3_file:        str,
            bucket_name:    str = None,
            show_log:       bool = True,
    ) -> List[str]:

        file_list = list()
        try:
            if bucket_name is None:
                bucket_name = self.bucket_name
            
            token = None
            while True:
                try:
                    params = {
                        'Bucket': bucket_name,
                        'Prefix': s3_file
                    }
                    token and params.update({'ContinuationToken': token})

                    response = self._client_s3.list_objects_v2(**params)
                    for content in response.get('Contents', []):
                        file_list.append(content.get('Key'))
                        
                    if not response.get('IsTruncated'):
                        break

                    token = response.get('NextContinuationToken')

                except (Exception,) as error:
                    st.get_error_exception(error, class_name=self.__class__.__name__)
                    
            show_log and st.LOGGING_SESSION.info(
                    f'--- Get file by Prefix [{len(file_list)} files] is success!'
            )

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return file_list
    
    def get_object(
            self,
            file_path: str
    ) -> Any:
        
        data = None
        try:
            response = self._client_s3.get_object(
                    Bucket=self.bucket_name,
                    Key=file_path
            )
            content = response['Body'].read().decode('utf-8')
            data = json.loads(content)
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return data
    
    def tagging_file(
            self,
            s3_file:        str,
            tag:            dict,
            bucket_name:    str = None,
            debug_mode:     bool = cf.DEBUG_MODE,
            show_log:       bool = True
    ) -> bool:

        is_success = False
        try:
            if debug_mode:
                return True

            if bucket_name is None:
                bucket_name = self.bucket_name
            
            is_success = self.check_tag_exist_s3_file(
                    tag=tag,
                    bucket_name=bucket_name,
                    s3_file=s3_file
            )
            if is_success:
                (show_log and
                 st.LOGGING_SESSION.warning(f'--- Tagging file {s3_file} is failed, b/c the tag already exists!'))
                return is_success
                
            self._client_s3.put_object_tagging(
                Bucket=bucket_name,
                Key=s3_file,
                Tagging=tag,
            )
            is_success = True
            show_log and st.LOGGING_SESSION.info(f'--- Tagging file {s3_file} is success!')

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return is_success
    
    def download_s3_folder(
            self,
            **kwargs
    ) -> None:
        try:
            check_key = all([key in kwargs.keys() for key in ['dirPath', 'objectName', 'bucketName']])
            if check_key:
                st.LOGGING_SESSION.info(f'Start downloading file in {kwargs["objectName"]}')

                file_list = self.get_files_by_prefix(**kwargs)
                for file_path in file_list:
                    if file_path[:-1] == kwargs['objectName']:
                        continue

                    file_name = os.path.join(kwargs['dirPath'], file_path)
                    os.makedirs(os.path.dirname(file_name), exist_ok=True)

                    self._client_s3.download_file(kwargs['bucketName'], file_path, file_name)
                    st.LOGGING_SESSION.info(f'Download file {file_path}')

                st.LOGGING_SESSION.info(f'Finish downloading file in {kwargs["objectName"]}')

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    def check_prefix_key_exist(
            self,
            s3_file:        str,
            bucket_name:    str = None
    ) -> bool:
        is_success = False
        try:
            count = 0
            while count < self.RETRY:
                try:
                    if bucket_name is None:
                        bucket_name = self.bucket_name
                        
                    self._client_s3.head_object(
                            Bucket=bucket_name,
                            Key=s3_file
                    )
                    is_success = True
                    break

                except NoCredentialsError:
                    self._client_s3 = self._create_client_session()
                    st.write_error_log(
                        error=f'S3 - No AWS credentials found - RETRY CREATE NEW CLIENT - {count}.',
                        class_name=self.__class__.__name__
                    )
                    is_success = False

                except (Exception,) as error:
                    if error.response['Error']['Code'] == '404':
                        is_success = False
                        break
                    else:
                        st.get_error_exception(error, class_name=self.__class__.__name__)

                count += 1

        except (Exception,) as error:
            if error.response['Error']['Code'] == '404':
                is_success = False
            else:
                st.write_error_log(error, class_name=self.__class__.__name__)

        return is_success

    def check_s3_permission(
            self,
            bucket_name: str
    ) -> bool:
        is_success = False
        try:
            self._client_s3.list_objects_v2(Bucket=bucket_name)
            is_success = True

        except NoCredentialsError:
            st.write_error_log(error="No AWS credentials found.", class_name=self.__class__.__name__)

        except PartialCredentialsError:
            st.write_error_log(error="Partial AWS credentials found. Please provide complete credentials.",
                               class_name=self.__class__.__name__)

        except ClientError as e:
            if e.response['Error']['Code'] == 'AccessDenied':
                msg = "Access to the S3 bucket is denied."
            else:
                msg = "An error occurred: " + str(e)
            st.write_error_log(error=msg, class_name=self.__class__.__name__)

        except (Exception, ) as error:
            st.write_error_log(error=error, class_name=self.__class__.__name__)

        return is_success
    
    @staticmethod
    def check_aws_profile_exists(
            profile_name: str
    ) -> bool:
        status = False
        try:
            session = boto3.Session(profile_name=profile_name)
            session.get_credentials()
            status = True
        
        except ProfileNotFound:
            st.LOGGING_SESSION.info(f"The profile '{profile_name}' does not exist.")
        
        return status

    def check_tag_exist_s3_file(
            self,
            tag:            dict,
            s3_file:        str,
            bucket_name:    str = None,
            show_log:       bool = True
    ) -> bool:
        
        """
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/get_object_tagging.html
        """

        status = False
        try:
            if bucket_name is None:
                bucket_name = self.bucket_name
            
            status = self.check_prefix_key_exist(
                    s3_file=s3_file,
                    bucket_name=bucket_name
            )
            if not status:
                (show_log
                 and st.LOGGING_SESSION.warning(f'File {s3_file} does not exist.', class_name=self.__class__.__name__))
                return status
                
            tag_files = self._client_s3.get_object_tagging(
                Bucket=bucket_name,
                Key=s3_file
            )
            status = all(x in tag_files['TagSet'] for x in tag['TagSet'])
            pass

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return status
