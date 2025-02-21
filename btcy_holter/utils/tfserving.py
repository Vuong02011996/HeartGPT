from btcy_holter import *


class TFServing(
    ABC
):
    TF_WAITING_TIME:        Final[int] = cf.DEFINE_TF_WAITING_TIME
    TF_TIMEOUT:             Final[int] = 5 * df.SECOND_IN_MINUTE
    
    def __init__(
            self,
            model_spec:         sr.TFServerModelStructure,
            datastore:          dict,
            server:             str = None,
            timeout_sec:        int | float = 60.0,         # seconds
            is_process_event:   bool = False,
            
    ) -> None:
        try:
            if server is None:
                server = cf.DEFAULT_TF_SERVER_ID
                
            self._server:                   Final[Any] = server
            self.model_spec:                Final[Any] = model_spec
            self.datastore:                 Final[Any] = datastore
            self.is_process_event:          Final[Any] = is_process_event

            self._timeout_sec:              float = timeout_sec

            self.standard_event_length:     Final[int] = cf.STANDARD_EVENT_LENGTH
            self.max_sample_process:        Final[int] = cf.MAX_SAMPLE_PROCESS
            
            self._channel, self._stub, self._request = self._create_stub()

            self._tf_severing_exception = dict()
            self._tf_severing_exception['timeout']:              Final[int] = 5 * df.SECOND_IN_MINUTE
            self._tf_severing_exception['resource_exhausted']:   Final[int] = 30

            self.log_performance_time = dict()
            self.log_performance_time['cpu']: float = 0.0
            self.log_performance_time['gpu']: float = 0.0
            self.log_performance_time['gpu/step']:  float = 0.0
            self.log_performance_time['step']:      int = 0

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
            
    def _log(
            self,
            message: str
    ) -> None:
        
        try:
            st.LOGGING_SESSION.info(f'[{self.model_spec.title_name}] [{self.__class__.__name__}] - {message}')
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    def _create_stub(
            self
    ) -> Any:
        
        channel = grpc.insecure_channel(
            self._server,
            options=(('grpc.enable_http_proxy', 0),)
        )
        stub = prediction_service_pb2_grpc.PredictionServiceStub(
            channel
        )

        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_spec.model_name
        request.model_spec.signature_name = self.model_spec.signature_name

        return channel, stub, request

    # @df.timeit
    def make_and_send_grpc_request(
            self,
            data:           NDArray,
            output_type:    str = 'float'
    ) -> NDArray:
        """Builds and sends request to TensorFlow model server."""
        self._request.inputs[self.model_spec.input_name].CopyFrom(
            tf.make_tensor_proto(
                data,
                dtype=tf.float32,
                shape=data.shape
            ))

        count = 0
        while True:
            try:
                output = self._stub.Predict(self._request, self._timeout_sec).outputs[self.model_spec.output_name]
                output = np.array(output.float_val) if output_type == 'float' else np.array(output.int64_val)
                break

            except (Exception,) as error:
                try:
                    status_code = error.code()
                    status_details = str(error.details()).split('\n')
                    log = f'[{self.model_spec.title_name}] - {status_code}'
    
                    if status_code in [
                        grpc.StatusCode.UNAVAILABLE,
                        grpc.StatusCode.RESOURCE_EXHAUSTED,
                        grpc.StatusCode.DEADLINE_EXCEEDED
                    ]:
    
                        log += f' - Waiting for the TF server for {self.TF_WAITING_TIME}s'
                        if status_code == grpc.StatusCode.DEADLINE_EXCEEDED:
                            self._timeout_sec *= 2
                            log += f' - timeout: {self._timeout_sec}'
                            if self._timeout_sec >= self._tf_severing_exception['timeout']:
                                self._timeout_sec = df.SECOND_IN_MINUTE
                                st.get_error_exception(f'- {status_details}')
    
                        if status_code == grpc.StatusCode.RESOURCE_EXHAUSTED:
                            count += 1
                            log += f' - count: {count}'
                            if count >= self._tf_severing_exception['resource_exhausted']:
                                st.get_error_exception(f'- {status_details}')
    
                        self._log(log)
                        time.sleep(self.TF_WAITING_TIME)
    
                    else:
                        st.get_error_exception(f'- {status_details}')
                        
                except (Exception,) as error:
                    st.get_error_exception(error, class_name=self.__class__.__name__)

        return output

    def close_channel(
            self
    ) -> None:
        self._channel.close()

    def log_time(
            self,
            process_time: Dict
    ) -> None:
        log_time_str = ''
        for x in process_time.keys():
            log_time_str += f'[{x}: {round(process_time[x], 4):7}s] '
        self._log(log_time_str)

    def update_performance_time(
            self,
            step:               int,
            start_process_time: float,
            process_time:       Dict
    ):
        self.log_performance_time['step']       = step
        self.log_performance_time['cpu']        = df.get_time_process(start_process_time) - process_time['tfServer']
        self.log_performance_time['gpu']        = process_time['tfServer']
        self.log_performance_time['gpu/step']   = process_time['tfServer'] / step

    @abstractmethod
    def prediction(self, **kwargs) -> Any:
        pass
