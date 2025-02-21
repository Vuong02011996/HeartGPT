from btcy_holter import *


class SVDetection(
        pt.Algorithm
):
    
    SV_PATTERNS: Final[List] = [
        (df.HOLTER_SVES_RUN,            "SS(S)+"),
        (df.HOLTER_VES_RUN,             "VV(V)+"),
        
        (df.HOLTER_SVES_QUADRIGEMINY,   "((NNNS){2})(NNNS)+((((N){1,4}S)NNNS(NNNS)+)(NNNS)+)*"),
        (df.HOLTER_VES_QUADRIGEMINY,    "((NNNV){2})(NNNV)+((((N){1,4}V)NNNV(NNNV)+)(NNNV)+)*"),
        
        (df.HOLTER_SVES_TRIGEMINAL,     "(((NNS){2})(NNS)+((((N){1,3}S)NNS(NNS)+)(NNS)+)*)"),
        (df.HOLTER_VES_TRIGEMINAL,      "(((NNV){2})(NNV)+((((N){1,3}V)NNV(NNV)+)(NNV)+)*)"),
        
        (df.HOLTER_SVES_BIGEMINY,       "(((NS){2})(NS)+((((N){1,3}S)NS(NS)+)(NS)+)*)"),
        (df.HOLTER_VES_BIGEMINY,        "(((NV){2})(NV)+((((N){1,3}V)NV(NV)+)(NV)+)*)"),
        
        (df.HOLTER_SVES_COUPLET,        "SS"),
        (df.HOLTER_VES_COUPLET,         "VV"),
        
        (df.HOLTER_SINGLE_SVES,         "S"),
        (df.HOLTER_SINGLE_VES,          "V"),
    ]
    
    MARK_SYMBOL:    Final[str] = 'X'
    INVALID_SYMBOL: Final[str] = 'Y'
    
    def __init__(
            self,
            data_structure: sr.AIPredictionResult,
            is_hes_process: bool = False,
    ) -> None:
        try:
            super(SVDetection, self).__init__(
                    data_structure=data_structure,
                    is_hes_process=is_hes_process
            )
            self.sync_beat_type()
            self.symbols = deepcopy(self.data_structure.symbol)
            
        except (Exception,) as error:
            st.write_error_log(error=error, class_name=self.__class__.__name__)
    
    def _mark_invalid_region(
            self
    ):
        try:
            # region Ignore VT and SVT
            if not self.is_hes_process:
                index = np.flatnonzero(np.isin(
                        self.data_structure.rhythm,
                        [
                            cf.RHYTHMS_DATASTORE['classes']['SVT'],
                            cf.RHYTHMS_DATASTORE['classes']['VT']
                        ]
                ))
            else:
                index = np.flatnonzero(np.logical_or(
                        df.check_hes_event(self.data_structure.rhythm, df.HOLTER_SVT),
                        df.check_hes_event(self.data_structure.rhythm, df.HOLTER_VT)
                ))

            if len(index) > 0:
                self.symbols[index] = self.MARK_SYMBOL
            # endregion Ignore VT and SVT
            
            # region Ignore ARTIFACT
            if self.is_hes_process:
                index = df.check_hes_event(self.data_structure.rhythm, df.HOLTER_ARTIFACT)
            else:
                index = self.data_structure.rhythm == cf.RHYTHMS_DATASTORE['classes']['OTHER']
                
            index = np.flatnonzero(index)
            if len(index) > 0:
                self.symbols[index] = self.MARK_SYMBOL
            # endregion Ignore ARTIFACT

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
            
    def _run(
            self,
    ) -> NDArray:
        
        sv_events = np.zeros_like(self.data_structure.symbol, dtype=int)
        try:
            input_string = ''.join(list(self.symbols))
            for hes_id, pattern in self.SV_PATTERNS:
                matches = list(re.finditer(pattern, input_string))
                indices = [[match.start(), match.end() - 1] for (match) in matches]
                if len(indices) == 0:
                    continue
                
                for start, end in indices:
                    substring: str = input_string[start: end + 1]
                    substring: str = substring.replace(df.HolterSymbols.SVE.value, self.INVALID_SYMBOL)
                    substring: str = substring.replace(df.HolterSymbols.VE.value, self.INVALID_SYMBOL)
                    self.symbols[start: end + 1] = list(substring)
                    sv_events[start: end + 1] |= hes_id
                input_string = ''.join(self.symbols)
        
        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)
        
        return sv_events
    
    def process(
            self,
    ) -> NDArray:
        
        sv_events = np.zeros_like(self.data_structure.beat)
        try:
            if not np.any(np.isin(self.symbols, [df.HolterSymbols.SVE.value, df.HolterSymbols.VE.value])):
                return sv_events
            
            self._mark_invalid_region()
            if not np.any(np.isin(self.symbols, [df.HolterSymbols.SVE.value, df.HolterSymbols.VE.value])):
                return sv_events
            
            sv_events = self._run()
        
        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
        
        return sv_events
        