from btcy_holter import *


class Algorithm(
        ABC
):
    INVALID:    Final[int] = 0
    VALID:      Final[int] = 1
    
    def __init__(
            self,
            data_structure: sr.AIPredictionResult,
            record_config:  sr.RecordConfigurations = None,
            is_hes_process: bool                    = False,
    ) -> None:
        try:
            self.data_structure:    sr.AIPredictionResult           = data_structure
            self.record_config:     Final[sr.RecordConfigurations]  = record_config
            self.is_hes_process:    Final[bool]                     = is_hes_process
            
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
            
    def _convert_ivcd_to_normal(
            self,
            symbols: NDArray
    ) -> NDArray:
        try:
            ind_ivcd = np.flatnonzero(symbols == df.HolterSymbols.IVCD.value)
            if len(ind_ivcd) > 0:
                symbols[ind_ivcd] = df.HolterSymbols.N.value
                
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return symbols
    
    def sync_beat_type(
            self,
    ) -> None:
        try:
            check = np.any(np.in1d(self.data_structure.symbol, list(df.HES_TO_SYMBOL.keys())))
            if check:
                self.data_structure.symbol = np.array(df.convert_hes_beat_to_symbol(self.data_structure.symbol))
            
            self.data_structure.symbol = self._convert_ivcd_to_normal(self.data_structure.symbol)
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
    
    def validate_input_is_symbol(
            self,
            symbols: NDArray | List
    ) -> bool:
        check = False
        try:
            check = np.any(np.in1d(np.unique(symbols), df.VALID_BEAT_TYPE + df.INVALID_BEAT_TYPE))
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
            
        return check
    
    @abstractmethod
    def process(
            self
    ) -> Any:
        pass
