import os

import btcy_holter.config as cf
import btcy_holter.define as df
import btcy_holter.stream as st

from typing import List, Final
from s5cmdpy import S5CmdRunner


class HelperS5CMD:
    """
    Helper class for S5CMD
    """
    
    PREFIX:     Final[str] = 's3://'

    def __init__(
            self,
            bucket_name: str = cf.AWS_BUCKET
    ) -> None:
        try:
            self._bucket_name:  Final[str]          = bucket_name
            self._runner = S5CmdRunner()
            
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
            
    def _get_s3_path(
            self,
            path: str
    ) -> str:
        
        try:
            if self.PREFIX not in path and self._bucket_name in path:
                path = os.path.join(self.PREFIX, path)
            
            elif self.PREFIX in path and self._bucket_name not in path:
                path = os.path.join(self.PREFIX, self._bucket_name, path.replace(self.PREFIX, ''))
                
            elif not all(x in path for x in [self.PREFIX, self._bucket_name]):
                path = os.path.join(self.PREFIX, self._bucket_name, path)
            
            elif self._bucket_name not in path:
                path = os.path.join(self.PREFIX, self._bucket_name, path)
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return path
    
    def download_files(
            self,
            s3_files:                List[str] | str,
            dest_dir:               str,
            show_log:               bool = True,
    ) -> bool:

        is_success = False
        try:
            os.makedirs(dest_dir, exist_ok=True)
            if isinstance(s3_files, str):
                s3_files = [s3_files]
                
            if len(s3_files) == 0:
                return is_success
            
            # region check file exists
            s3_files = list(filter(
                    lambda x: not df.check_file_exists(os.path.join(dest_dir, os.path.basename(x))),
                    s3_files
            ))
            if len(s3_files) == 0:
                is_success = True
                return is_success
            # endregion check file exists
            
            self._runner.download_from_s3_list(
                    s3_uris=list(map(self._get_s3_path, s3_files)),
                    dest_dir=dest_dir,
                    simplified_print=True
            )
            
            is_success = True
            show_log and st.LOGGING_SESSION.info(f'---- Download: {len(s3_files)} / {len(os.listdir(dest_dir))}')
            
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return is_success
    
    def upload_files(
            self,
            files:      List[str] | str,
            s3_dir:     str,
            show_log:   bool = True,
    ) -> bool:
        is_success = False
        try:
            if cf.DEBUG_MODE:
                is_success = True
                return is_success
            
            if isinstance(files, str):
                files = [files]
            
            if len(files) == 0:
                is_success = True
                show_log and st.LOGGING_SESSION.info(f'---- No files to upload')
                
                return is_success
            
            files = list(filter(
                    lambda x: df.check_file_exists(x),
                    files
            ))
            if len(files) == 0:
                is_success = True
                show_log and st.LOGGING_SESSION.info(f'---- No files to upload')
                
                return is_success
            
            s3_dir = self._get_s3_path(s3_dir)
            if not s3_dir.endswith('/'):
                s3_dir += '/'
                
            for file in files:
                self._runner.cp(
                        from_str=file,
                        to_str=s3_dir,
                        simplified_print=True
                )
            
            show_log and st.LOGGING_SESSION.info(f'---- Upload: {files}')
            is_success = True

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return is_success

    def upload_folder(
            self,
            s3_dir:     str,
            local_dir:  str,
            show_log:   bool = True,
    ) -> bool:
        is_success = False
        try:
            if cf.DEBUG_MODE:
                is_success = True
                return is_success
            
            if not os.path.exists(local_dir):
                st.LOGGING_SESSION.error(f'Folder {local_dir} is not exists!')
                return is_success
            
            total_files = len(os.listdir(local_dir))
            if total_files == 0:
                return is_success
            
            s3_paths = self._get_s3_path(s3_dir)
            if not s3_paths.endswith('/'):
                s3_paths += '/'
            
            self._runner.sync(
                    destination=s3_dir,
                    source=local_dir,
                    simplified_print=True
            )
            
            show_log and st.LOGGING_SESSION.info(f'---- Upload: {total_files}')

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    def get_files_from_s3_prefix(
            self,
            s3_dir: str
    ) -> List:

        result = list()
        try:
            result = list(
                    self._runner.ls(
                            s3_uri=self._get_s3_path(s3_dir)
                    )
            )

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return result

    def check_file_exists(
            self,
            s3_file: str
    ) -> bool:

        is_success = False
        try:
            files = list(
                    self._runner.ls(
                            s3_uri=self._get_s3_path(s3_file)
                    )
            )
            is_success = len(files) > 0

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return is_success
