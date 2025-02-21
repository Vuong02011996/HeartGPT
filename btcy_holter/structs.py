from typing import Any


class AlgorithmConfigurations:
    """
    Class to handle configurations for various algorithms.
    """
    
    def __init__(
            self
    ) -> None:
        """
        Class to handle configurations for various algorithms.
        """
        
        self.run_pqst_detection:        bool = True
        self.run_noise_detection:       bool = True
        self.run_hes_classification:    bool = True
        
        self.run_sv_detection:          bool = True
        self.run_pause_detection:       bool = True
        self.run_tachy_brady_detection: bool = True
        
    def get_default_configurations(
            self,
            **kwargs
    ) -> Any:
        """
        Sets the default configurations for the algorithms.

        Args:
            **kwargs: Arbitrary keyword arguments to override default values.

        Returns:
            self: The instance with updated configurations.
        """
        
        self.run_pqst_detection:        bool = kwargs.get('run_pqst_detection', True)
        
        self.run_noise_detection:       bool = kwargs.get('run_noise_detection', True)
        self.run_hes_classification:    bool = kwargs.get('run_hes_classification', True)
        
        self.run_sv_detection:          bool = kwargs.get('run_sv_detection', True)
        self.run_pause_detection:       bool = kwargs.get('run_pause_detection', True)
        self.run_tachy_brady_detection: bool = kwargs.get('run_tachy_brady_detection', True)
        
        return self
    

class RecordConfigurations:
    """
    Class to handle configurations for recording.
    """
    
    def __init__(
            self
    ) -> None:
        """
        Initializes the RecordConfigurations with default values.
        """
        
        self.tachy_threshold:       Any = None
        self.brady_threshold:       Any = None
        self.pause_threshold:       Any = None
        
        self.gain:                  Any = None
        self.sampling_rate:         Any = None
        self.number_of_channels:    Any = None
        
        self.channel:               Any = None
        self.timezone:              Any = None
        
        self.record_id:             Any = None
        self.record_path:           Any = None
        self.record_file_index:     Any = None
        self.record_start_time:     Any = None
        self.record_stop_time:      Any = None
    
    @staticmethod
    def check_all_variables_values(
            record_config: dict
    ) -> bool:
        """
        Checks if all variables in the given record configuration are not None.

        Args:
            record_config: The record configuration instance to check.

        Returns:
            bool: True if all variables are not None, False otherwise.
        """
        return all(value is not None for value in record_config.__dict__.values())
        
    def is_none(
            self
    ) -> bool:
        """
        Checks if all variables in the instance are None.

        Returns:
            bool: True if all variables are None, False otherwise.
        """
        return all(value is None for value in self.__dict__.values())
    
    def is_not_none(
            self
    ) -> bool:
        """
        Checks if any variable in the instance is not None.

        Returns:
            bool: True if any variable is not None, False otherwise.
        """
        return any(value is not None for value in self.__dict__.values())
    
    def is_channel_none(
            self
    ) -> bool:
        """
        Checks if the channel variable is None.

        Returns:
            bool: True if the channel is None, False otherwise.
        """
        return self.channel is None
    
    def is_study_name_none(
            self
    ) -> bool:
        """
        Checks if the record_path variable is None.

        Returns:
            bool: True if the record_path is None, False otherwise.
        """
        return self.record_path is None
    
    def get_default_configurations(
            self,
            record_path:    str = None,
            channel:        int = None,
            **kwargs
    ) -> Any:
        """
        Sets the default configurations for the recording.

        Args:
            record_path (str, optional): The path to the record.
            channel (int, optional): The channel number.
            **kwargs: Arbitrary keyword arguments to override default values.

        Returns:
            RecordConfigurations: The instance with updated configurations.
        """
        
        from btcy_holter import cf, df
        self.tachy_threshold = kwargs.get('tachy_threshold', cf.TACHY_THR)
        self.brady_threshold = kwargs.get('brady_threshold', cf.BRADY_THR)
        self.pause_threshold = kwargs.get('pause_threshold', cf.PAUSE_RR_THR)
        
        if record_path is not None:
            self.record_path = record_path
        
        if channel is not None and df.is_channel_valid(channel):
            self.channel = channel
            
        self.timezone:              str = kwargs.get('timezone', cf.TIMEZONE)
        self.gain:                  float = kwargs.get('gain', cf.GAIN)
        self.sampling_rate:         int = kwargs.get('sampling_rate', cf.SAMPLING_RATE)
        self.number_of_channels:    int = kwargs.get('number_of_channels', cf.NUMBER_OF_CHANNELS)
        
        self.record_file_index:     int = 0
        
        return self
    
    
class AIPredictionResult:
    """
    Class to handle AI prediction results.
    """
    
    def __init__(
            self
    ) -> None:
        """
        Initializes the AIPredictionResult with default values.
        """
        
        self.channel:           Any = None
        self.ecg_signal:        Any = None
        self.sampling_rate:     Any = None
                                
        self.beat:              Any = list()
        self.symbol:            Any = list()
        self.rhythm:            Any = list()
        self.beat_channel:      Any = list()

        self.lead_off:          Any = list()
        self.lead_off_frames:   Any = list()


class AINumPyResult:
    """
    Initializes the AIPredictionResult with default values.
    """
    
    def __init__(
            self
    ) -> None:
        """
        Initializes the AINumPyResult with default values.
        """
        
        self.epoch:             Any = list()
        self.beat:              Any = list()
        self.beat_types:        Any = list()
        self.events:            Any = list()
        self.beat_channels:     Any = list()
        self.rr_cell_ids:       Any = list()
        self.pvc_morphology:    Any = list()


class TFServerModelStructure:
    """
    Class to handle TensorFlow server model structure.
    """
    
    def __init__(
            self,
            model_name:         str,
            signature_name:     str,
            input_name:         str,
            output_name:        str,
            classes:            Any = None
    ) -> None:
        """
        Initializes the TFServerModelStructure with the given parameters.

        Args:
            model_name (str):           The name of the model.
            signature_name (str):       The signature name of the model.
            input_name (str):           The input name for the model.
            output_name (str):          The output name for the model.
            classes (Any, optional):    The classes for the model.
        """
        
        self.model_name:        Any = model_name
        self.signature_name:    Any = signature_name
        self.input_name:        Any = input_name
        self.output_name:       Any = output_name
        self.classes:           Any = classes
        self.title_name:        Any = f'{model_name.upper()} NET'
