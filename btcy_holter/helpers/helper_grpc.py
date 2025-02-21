from btcy_holter import *


class HelperGrpc(
    object
):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            service_url:    str,
            service_name:   str = 'AiApi',
            use_tls:        bool = False
    ) -> None:
        try:

            self.proto_path:        Final[str] = join(dirname(abspath(sys.modules['__main__'].__file__)), 'proto')

            self.service_name:      Final[str] = service_name
            self.service_url:       Final[str] = service_url
            self.is_tls:            Final[bool] = use_tls

            self._create_stub()

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

    def _create_stub(
            self
    ) -> None:
        try:
            self._create_pb2_file()
            self._load_pb2_file()

            if self.is_tls:
                self.channel = grpc.secure_channel(self.service_url, grpc.ssl_channel_credentials())
            else:
                self.channel = grpc.insecure_channel(self.service_url)

            self.stub = getattr(self.service_pb2_grpc, f'{self.service_name}Stub')(self.channel)

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

    def _create_pb2_file(
            self,
    ) -> None:
        try:
            list_proto_files = glob.glob(f'{self.proto_path}/*.proto')
            for proto_file in list_proto_files:
                if self.service_name in basename(proto_file):
                    protoc.main([
                        f'-I={self.proto_path}',
                        f'--proto_path={self.proto_path}',
                        f'--python_out={self.proto_path}',
                        f'--grpc_python_out={self.proto_path}',
                        proto_file,
                    ])

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

    def _load_pb2_file(
            self,
    ) -> None:
        try:
            pb2_name = f'{self.service_name}_pb2'
            pb2_file = join(self.proto_path, f'{pb2_name}.py')

            pb2_grpc_name = f'{self.service_name}_pb2_grpc'
            pb2_grpc_file = join(self.proto_path, f'{pb2_grpc_name}.py')

            self.service_pb2        = SourceFileLoader(pb2_name, pb2_file).load_module()
            self.service_pb2_grpc   = SourceFileLoader(pb2_grpc_name, pb2_grpc_file).load_module()

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
