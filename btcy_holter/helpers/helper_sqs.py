import boto3

import btcy_holter.stream as st
import btcy_holter.config as cf
from typing import Any, Final


class HelperSqs(
        object
):
    NAME: Final[str] = 'sqs'
    
    def __init__(
            self,
            queue_url:          str,
            response_queue_url: str,
    ) -> None:
        try:
            self._client_sqs:           Any = self._create_client_session()

            self.is_stop:               bool = False
            self.queue_url:             str = queue_url
            self.response_queue_url:    str = response_queue_url

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

    def _create_client_session(
            self
    ):
        client = None
        try:
            st.LOGGING_SESSION.info(f'[{self.NAME}] - [use IRSA: {cf.USE_IRSA}] - aws profile: {cf.AWS_PROFILE}')
            if cf.DEBUG_MODE or cf.USE_IRSA:
                session = boto3.session.Session(
                        profile_name=cf.AWS_PROFILE
                )
                
                client = session.client(
                        self.NAME,
                        region_name=cf.AWS_REGION
                )
                
            else:
                client = boto3.client(
                        self.NAME,
                        region_name=cf.AWS_REGION
                )

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return client

    def __poll_sqs_message(
            self,
            callback,
            include_empty_msg: bool = False
    ) -> None:
        try:
            while True:
                if self.is_stop:
                    break
    
                sqs_message = self.__receive_sqs_message()
                if include_empty_msg or sqs_message:
                    callback(sqs_message)
    
            st.LOGGING_SESSION.info('Stop polling SQS.')
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    def __receive_sqs_message(
            self
    ) -> Any:
        
        sqs_message = None
        try:
            sqs_response = self._client_sqs.receive_message(
                QueueUrl=self.queue_url,
                AttributeNames=[
                    'MessageGroupId',
                    'ApproximateReceiveCount'
                ],
            )
            if 'Messages' not in sqs_response.keys():
                return sqs_message
    
            message = sqs_response['Messages']
            if not (isinstance(message, list) and len(message) > 0):
                return sqs_message
            
            sqs_message = sqs_response['Messages'][0]
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
            
        return sqs_message

    def delete_sqs_message(
            self,
            message: dict
    ) -> None:
        try:
            self._client_sqs.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=message['ReceiptHandle']
            )
            st.LOGGING_SESSION.info('Deleted message')

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    def send_result_message(
            self,
            message,
            message_group_id
    ) -> None:
        try:
            self._client_sqs.send_message(
                QueueUrl=self.response_queue_url,
                MessageBody=message,
                MessageGroupId=message_group_id
            )
            
        except (Exception,) as error:
            st.write_error_log(error=error, class_name=self.__class__.__name__)
            
    def change_sqs_message_visibility(
            self,
            message: dict,
            visibility_timeout: int = 1
    ) -> None:
        try:
            self._client_sqs.change_message_visibility(
                QueueUrl=self.queue_url,
                ReceiptHandle=message['ReceiptHandle'],
                VisibilityTimeout=visibility_timeout
            )
            st.LOGGING_SESSION.info('Changed message visibility')

        except (Exception,) as error:
            st.write_error_log(error=error, class_name=self.__class__.__name__)

    def start(
            self,
            callback,
            **kwargs,
    ) -> None:
        try:
            st.LOGGING_SESSION.info('Start polling sqs message')
            self.__poll_sqs_message(
                    callback=callback,
                    include_empty_msg=kwargs.get('include_empty_msg', False)
            )
        
        except (Exception,) as error:
            st.write_error_log(error=error, class_name=self.__class__.__name__)

    def stop(
            self
    ) -> None:
        try:
            self.is_stop = True
            
        except (Exception,) as error:
            st.write_error_log(error=error, class_name=self.__class__.__name__)