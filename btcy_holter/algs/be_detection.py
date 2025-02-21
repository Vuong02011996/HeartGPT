from btcy_holter import *

from btcy_holter.algs.sve_ve_detection import (
    SVDetection
)

from btcy_holter.algs.pause_detection import (
    PausedDetection
)

from btcy_holter.algs.tachy_brady_detection import (
    TachyBradyDetection
)


class BEDetection:
    
    def __init__(
            self,
            data_structure:             sr.AIPredictionResult,
            record_config:              sr.RecordConfigurations,
            algorithm_config:           sr.AlgorithmConfigurations = None,
            is_hes_process:             bool = False,
    ) -> None:
        try:
            self.record_config:                 Final[Any] = record_config
            self.algorithm_config:              Final[Any] = algorithm_config
            
            self.is_hes_process:                Final[bool] = is_hes_process
            self.rhythm_class:                  Final[Dict] = cf.RHYTHMS_DATASTORE['classes']
            
            if self.algorithm_config is None:
                self.algorithm_config = sr.AlgorithmConfigurations()
                self.algorithm_config = self.algorithm_config.get_default_configurations()
            
            self.data_structure = data_structure
            if self.is_hes_process:
                self.data_structure.symbol = df.convert_hes_beat_to_symbol(self.data_structure.symbol)
            
            ind_ivcd = np.flatnonzero(self.data_structure.symbol == df.HolterSymbols.IVCD.value)
            if len(ind_ivcd) > 0:
                self.data_structure.symbol[ind_ivcd] = df.HolterSymbols.N.value
            
            self.beat_events = np.zeros_like(self.data_structure.beat)
        
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
            
    def _run_sv(
            self,
    ) -> None:
        
        try:
            sv_func = SVDetection(
                    data_structure=self.data_structure,
                    is_hes_process=self.is_hes_process
            )
            self.beat_events |= sv_func.process()
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    # @df.timeit
    def _run_tachy_brady_detection(
            self,
    ) -> None:
        try:
            tb_function = TachyBradyDetection(
                    data_structure=self.data_structure,
                    record_config=self.record_config,
                    is_hes_process=self.is_hes_process
            )
            self.beat_events |= tb_function.process()
            pass
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    def _run_pause_detection(
            self,
    ) -> None:
        try:
            pause_function = PausedDetection(
                data_structure=self.data_structure,
                record_config=self.record_config,
            )
            self.beat_events |= pause_function.process()

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

    # @df.timeit
    def process(
            self,
    ) -> [NDArray, NDArray]:

        try:
            self.algorithm_config.run_sv_detection and self._run_sv()
            self.algorithm_config.run_pause_detection and self._run_pause_detection()
            self.algorithm_config.run_tachy_brady_detection and self._run_tachy_brady_detection()

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return self.beat_events
