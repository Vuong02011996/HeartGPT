from btcy_holter import *
from btcy_holter.helpers import HelperS3, HelperS5CMD


S3_HELPER = HelperS3()
S5_HELPER = HelperS5CMD()


# @df.timeit
def upload_file_to_aws_s3(
        local_file:  str,
        s3_file:     str,
        show_log:    bool = True,
        method:      str = 's5cmd',
) -> bool:
    
    status = False
    try:
        match method:
            case 's5cmd':
                status = S5_HELPER.upload_files(
                    files=local_file,
                    s3_dir=dirname(s3_file),
                    show_log=show_log
                )
                
            case 'boto3':
                status = S3_HELPER.upload_file(
                    local_file=local_file,
                    s3_file=s3_file,
                    show_log=show_log
                )
            
    except (Exception,) as error:
        st.write_error_log(error)

    return status


# @df.timeit
def download_file_from_aws_s3(
        s3_file:    str,
        local_file: str,
        show_log:   bool = True,
        method:     str = 's5cmd',
) -> bool:
    status = False
    try:
        if local_file is None:
            return status

        match method:
            case 's5cmd':
                status = S5_HELPER.download_files(
                        s3_files=s3_file,
                        dest_dir=dirname(local_file),
                        show_log=show_log
                )
            
            case 'boto3':
                status = S3_HELPER.download_file(
                        local_file=local_file,
                        s3_file=s3_file,
                        show_log=show_log
                )
            
    except (Exception,) as error:
        st.write_error_log(error)
    
    return status


def download_files_by_s5cmd(
        s3_files:   List[str],
        dest_dir:   str,
        show_log:   bool = True
) -> bool:
    status = False
    try:
        status = S5_HELPER.download_files(
                s3_files=s3_files,
                dest_dir=dest_dir,
                show_log=show_log
        )
        
    except (Exception,) as error:
        st.write_error_log(error)
    
    return status


def upload_folder_by_s5cmd(
        s3_dir:         str,
        local_dir:      str,
        show_log:       bool = True
) -> bool:
    status = False
    try:
        status = S5_HELPER.upload_folder(
                s3_dir=s3_dir,
                local_dir=local_dir,
                show_log=show_log
        )
    
    except (Exception,) as error:
        st.write_error_log(error)
    
    return status


def upload_files_by_s5cmd(
        s3_dir:     str,
        files:      List[str] | str,
        show_log:   bool = True
) -> bool:
    status = False
    try:
        status = S5_HELPER.upload_files(
                s3_dir=s3_dir,
                files=files,
                show_log=show_log
        )
    
    except (Exception,) as error:
        st.write_error_log(error)
    
    return status


# @df.timeit
def tagging_file_in_aws_s3(
        s3_file:    str,
        tags:       List,
        show_log:   bool = True
) -> bool:
    
    status = False
    try:
        status = S3_HELPER.tagging_file(
            s3_file=s3_file,
            tag={'TagSet': tags},
            show_log=show_log
        )
        
    except (Exception,) as error:
        st.write_error_log(error)
        
    return status


# @df.timeit
def check_s3_file_exists(
        s3_file:    str,
        show_log:   bool = False,
        method:     str = 's5cmd'
) -> bool:
    
    status = False
    try:
        match method:
            case 's5cmd':
                status = S5_HELPER.check_file_exists(
                    s3_file=s3_file
                )
                
            case 'boto3':
                status = S3_HELPER.check_prefix_key_exist(
                    s3_file=s3_file
                )
            
        (show_log
         and st.LOGGING_SESSION.warning(f'Check {s3_file} is {"not exists" if not status else "exists"}!'))
        
    except (Exception,) as error:
        st.write_error_log(error)
    
    return status


def get_all_files_in_s3_bucket(
        s3_path:        str,
        show_log:       bool = False,
        method:         str = 's5cmd'
) -> List[str]:
    
    files = list()
    try:
        match method:
            case 's5cmd':
                files = S5_HELPER.get_files_from_s3_prefix(
                    s3_dir=s3_path
                )
                
            case 'boto3':
                files = S3_HELPER.get_files_by_prefix(
                    s3_file=s3_path
                )
        
        (show_log
         and st.LOGGING_SESSION.warning(f'Total files is {len(files)}!'))
        
    except (Exception,) as error:
        st.write_error_log(error)
    
    return files
