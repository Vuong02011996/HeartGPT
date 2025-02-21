from btcy_holter import *


class RRHeatMap:
    # region param
    HM_TYPE: Final[Dict] = {
        'INVALID':      0,
        'N':            1,
        'S':            2,
        'V':            3,
        'qToQ':         4,
        'qToNSV':       5,
        # 'nsvToQ': 6
    }
    HM_TYPE_REVERT:     Final[Dict] = {i: k for k, i in HM_TYPE.items()}
    
    CELL_ID:            Final[int] = 10 ** 4
    # endregion param

    # @df.timeit
    def __init__(
            self,
            beats:          Union[NDArray, pl.Series, List],
            beat_types:     Union[NDArray, pl.Series, List],
            events:         Union[NDArray, pl.Series, List],
            file_index:     Union[NDArray, pl.Series, List],
            sampling_rate:  int,
            method:         str = 'RR-Ratio',
            **kwargs
    ) -> None:
        try:
            self.method:            Final[str] = method
            self.sampling_rate:     Final[int] = sampling_rate

            self.beats:             Final[NDArray] = self.init(beats)
            self.beat_types:        Final[NDArray] = self.init(beat_types, is_copy=True)
            self.events:            Final[NDArray] = self.init(events)
            self.file_index:        Final[NDArray] = self.init(file_index)

            self.ox, self.oy = self._get_oxy_config(**kwargs)
            self.valid_idx, self.invalid_idx = self._get_index_valid()

            self.rr_x_values, self.rr_y_values = self.calculate_rr_values()
            
            self.rr_cell_ids = np.zeros_like(self.beats, dtype=int)
            self.rr_cell_reviewed = np.zeros_like(self.beats, dtype=int)

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

    def _log(
            self,
    ):
        try:
            for _type, _id in self.HM_TYPE.items():
                cell_ind = np.flatnonzero(self.rr_cell_ids // self.CELL_ID == _id)
                if len(cell_ind) == 0:
                    continue

                st.LOGGING_SESSION.info(f'+ {_type}')
                cells = dict(Counter(self.rr_cell_ids[cell_ind]))
                keys = list(sorted(cells.keys()))

                for x in range(0, len(keys), 5):
                    tmp = [f'{keys[x + i]:6}: {cells[keys[x + i]]:6}' for i in range(5) if x + i < len(keys)]
                    st.LOGGING_SESSION.info('\t {}'.format(',\t'.join(tmp)))

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

    def _review(
            self
    ):
        try:
            total_axis = deepcopy(list(self.HM_TYPE.keys()))
            total_axis.remove('INVALID')

            fig, axis = plt.subplots(2, 3, sharex=False, sharey=False)
            fig.subplots_adjust(hspace=0.25, wspace=0.2, left=0.02, right=0.99, bottom=0.05, top=0.95)

            for idx, (ax) in enumerate(axis.flatten()):
                if idx > len(total_axis) - 1:
                    continue

                ax.set_frame_on(False)
                if total_axis[idx] == 'INVALID' or idx > len(total_axis):
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                ax.set_title(total_axis[idx], fontdict={'fontsize': 12, 'fontweight': 'bold', 'fontstyle': 'italic'})
                index = np.flatnonzero(self.rr_cell_ids // self.CELL_ID == self.HM_TYPE[total_axis[idx]])
                if len(index) == 0:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                ids = dict(Counter(self.rr_cell_ids[index]))
                oxy = np.zeros((len(self.ox), len(self.oy)), dtype=int)
                for i in range(len(self.oy)):
                    for j in range(len(self.ox)):
                        cell_id = self.__get_cell_id(total_axis[idx], i, j)
                        count = ids.get(cell_id, 0)
                        if count > 0:
                            oxy[i, j] = count
                            ax.text(j + 0.5, i + 0.5, cell_id, ha='center', va='center', color='red', fontsize=7)

                heatmap = ax.pcolor(np.log(oxy), cmap='plasma', alpha=0.8)
                plt.colorbar(heatmap, ax=ax)

                ax.set_ylabel('RR Ratio')
                ax.set_xlabel('RR Intervals (ms)')

                ax.grid(True, alpha=0.5)
                ax.set_xticks(np.arange(len(self.ox)) + 0.5, self.ox, rotation=45, fontsize=5)
                ax.set_yticks(np.arange(len(self.oy)) + 0.5, self.oy, fontsize=5)

                ax.set_xlim(0, len(self.ox) + 1)
                ax.set_ylim(0, len(self.oy) + 1)

            plt.show()
            plt.close()

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
            pass

    def init(
            self,
            x:          Union[NDArray, pl.Series, List],
            is_copy:    bool = False
    ) -> NDArray:
        try:
            if isinstance(x, pl.Series):
                if is_copy:
                    return x.to_numpy().copy()

                return x.to_numpy()

            if isinstance(x, list):
                return np.array(x)

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)

        return x

    # @df.timeit
    def _get_oxy_config(
            self,
            **kwargs
    ) -> [NDArray, NDArray]:

        ox, oy = np.array([]), np.array([])
        try:

            try:
                oy_range = kwargs['ratio']
            except (Exception,):
                oy_range = [0, 4.0]

            try:
                oy_step = kwargs['step_ratio']
            except (Exception,):
                oy_step = 0.1

            try:
                ox_range = kwargs['duration']
            except (Exception,):
                ox_range = [0, 4000]  # ms

            try:
                ox_step = kwargs['step_duration']
            except (Exception,):
                ox_step = 100  # ms

            if any(x is None for x in [oy_range, oy_step, ox_range, ox_step]):
                st.get_error_exception('invalid config param.')

            ox_range[-1] = ox_range[-1] + ox_step
            oy_range[-1] = oy_range[-1] + oy_step

            ox = np.arange(*ox_range, step=ox_step)
            oy = np.round(np.arange(*oy_range, step=oy_step), 1)

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return ox, oy

    # @df.timeit
    def _get_index_valid(
            self
    ) -> [NDArray, NDArray]:
        """
            - Disregard beats occurring within a continuous two-hour period.
            - Exclude beats within the Artifact region and the two beats adjacent to it.
        """

        valid_index = np.array([])
        invalid_index = np.array([])
        try:
            index_cons = np.flatnonzero(np.diff(self.file_index) != 0)

            index_marked_beats = np.flatnonzero(self.beat_types == df.HolterBeatTypes.MARKED.value)

            index_artifact = df.get_index_artifact_in_region(self.events)
            index_artifact = np.unique((index_artifact.reshape((-1, 1)) + np.arange(-1, 2)).flatten())

            ignore_index = np.concatenate([
                np.array([0]),
                index_cons,
                index_cons + 1,
                index_artifact,
                index_marked_beats,
                [len(self.beats) - 1]
            ])
            ignore_index = np.sort(np.unique(ignore_index.flatten()))
            check = np.flatnonzero(ignore_index == -1)
            if len(check) > 0:
                ignore_index = np.delete(ignore_index, check)
                ignore_index = np.concatenate([ignore_index, [len(self.beats) - 1]])

            valid_index = np.arange(len(self.beats))
            if len(ignore_index) > 0:
                inv = np.flatnonzero(np.logical_and(ignore_index >= 0, ignore_index < len(self.beats)))
                invalid_index = ignore_index[inv]
                valid_index = np.delete(valid_index, invalid_index)

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return valid_index, invalid_index

    # @df.timeit
    def calculate_rr_values(
            self,
    ) -> [NDArray, NDArray]:

        rr_x_values, rr_y_values = np.array([]), np.array([])
        try:
            match self.method:
                case 'RR-Ratio':
                    rr = (np.diff(self.beats) / self.sampling_rate) * df.MILLISECOND
                    rr_x_values = np.concatenate((rr, [0]))
                    rr_y_values = np.concatenate(([0], rr[1:] / rr[:-1]))

                case _:
                    st.get_error_exception(
                        f'The present iteration of HeatMap does not provide support for the method: {self.method}'
                    )

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return rr_x_values, rr_y_values

    def get_hm_type(
            self,
            hm_type: str
    ) -> NDArray:

        idx = np.array([])
        try:
            match hm_type:
                case 'N':
                    idx = np.flatnonzero(self.beat_types[self.valid_idx] == df.HolterBeatTypes.N.value)

                case 'S':
                    idx = np.flatnonzero(self.beat_types[self.valid_idx] == df.HolterBeatTypes.SVE.value)

                case 'V':
                    idx = np.flatnonzero(self.beat_types[self.valid_idx] == df.HolterBeatTypes.VE.value)

                case 'qToQ':
                    idx = np.flatnonzero(np.logical_and(
                        self.beat_types[self.valid_idx] == df.HolterBeatTypes.OTHER.value,
                        self.beat_types[self.valid_idx + 1] == df.HolterBeatTypes.OTHER.value
                    ))

                case 'qToNSV':
                    idx = np.flatnonzero(np.logical_and(
                        self.beat_types[self.valid_idx] == df.HolterBeatTypes.OTHER.value,
                        self.beat_types[self.valid_idx + 1] != df.HolterBeatTypes.OTHER.value
                    ))

                case _:
                    pass

        except (Exception,) as error:
            st.get_error_exception(error, class_name=self.__class__.__name__)

        return idx

    def __get_index(
            self,
            index:      int,
            rrs:        NDArray,
            axes_type:  str = 'ox'
    ) -> NDArray:

        axes = self.ox if axes_type == 'ox' else self.oy
        if index == len(axes) - 1:
            ind_valid = np.flatnonzero(rrs >= axes[index])
        else:
            ind_valid = np.flatnonzero(np.logical_and(
                rrs >= axes[index],
                rrs < axes[index + 1]
            ))

        return ind_valid

    def __get_cell_id(
            self,
            cell_type: int | str,
            row_index: int,
            col_index: int,
    ) -> int:
        if isinstance(cell_type, str):
            cell_type = self.HM_TYPE[cell_type]

        return int(f'{cell_type}{row_index:02d}{col_index:02d}')

    # @df.timeit
    def _process_vertical_axes(
            self,
            _id:        int,
            row:        int,
            beat_index: NDArray
    ) -> None:

        try:
            for col in range(len(self.ox)):
                ind = self.__get_index(col, self.rr_x_values[beat_index], axes_type='ox')
                if len(ind) > 0:
                    self.rr_cell_ids[beat_index[ind]] = self.__get_cell_id(_id, row, col)
                    beat_index = np.delete(beat_index, ind)

                if len(beat_index) == 0:
                    break

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
            pass

    # @df.timeit
    def _process_horizontal_axes(
            self,
            _id:        int,
            beat_index: NDArray
    ) -> None:
        try:
            for row in range(len(self.oy)):
                ind = self.__get_index(row, self.rr_y_values[beat_index], axes_type='oy')
                if len(ind) > 0:
                    self._process_vertical_axes(_id, row, beat_index[ind])
                    beat_index = np.delete(beat_index, ind)

                if len(beat_index) == 0:
                    break

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
            pass

    # @df.timeit
    def _update_sv_beats_in_miss_rr_region(
            self
    ) -> None:
        try:
            for sym, beat_type in [
                [df.HolterSymbols.SVE.value, df.HolterBeatTypes.SVE.value],
                [df.HolterSymbols.VE.value, df.HolterBeatTypes.VE.value]
            ]:
                ind = np.flatnonzero(np.logical_and(
                    self.beat_types[self.invalid_idx] == beat_type,
                    ~df.check_hes_event(self.events[self.invalid_idx], df.HOLTER_ARTIFACT)
                ))
                if len(ind) == 0:
                    continue

                self.rr_cell_ids[self.invalid_idx[ind]] = self.__get_cell_id(
                        sym,
                        row_index=len(self.oy) - 1,
                        col_index=len(self.oy) - 1
                )

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
            pass

    # @df.timeit
    def _process_heatmap(
            self
    ) -> None:
        
        try:
            for _type, _id in self.HM_TYPE.items():
                if _type == 'INVALID':
                    continue

                index = self.get_hm_type(_type)
                if len(index) == 0:
                    continue

                self._process_horizontal_axes(_id, self.valid_idx[index])
                self.valid_idx = np.delete(self.valid_idx, index)

            self._update_sv_beats_in_miss_rr_region()

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
            pass

    # @df.timeit
    def _restore_reviewed_cells(
            self,
            cell_reviewed:  NDArray,
    ) -> None:
        try:
            if cell_reviewed is None:
                return
            
            index = np.flatnonzero(np.isin(self.rr_cell_ids, cell_reviewed))
            self.rr_cell_reviewed[index] = 1

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
            pass

    # @df.timeit
    def process(
            self,
            cell_reviewed:  NDArray = None,
            review:         bool = False,
            log:            bool = False
    ) -> [NDArray]:

        try:
            if len(self.valid_idx) == 0:
                return self.rr_cell_ids, self.rr_cell_reviewed

            self._process_heatmap()
            self._restore_reviewed_cells(cell_reviewed)
            
            review and self._review()
            log and self._log()

        except (Exception,) as error:
            st.write_error_log(error, class_name=self.__class__.__name__)
            pass
        
        return self.rr_cell_ids, self.rr_cell_reviewed
    