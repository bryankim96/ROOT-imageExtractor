from astropy import units as u
from astropy.coordinates import Angle
from ctapipe.io.eventsource import EventSource
from ctapipe.io.containers import DataContainer
from ctapipe.instrument import (
    TelescopeDescription,
    SubarrayDescription,
    OpticsDescription,
    CameraGeometry,
)

import numpy as np
import warnings


class DL1DHEventSource(EventSource):
    """
    EventSource for the dl1_data_handler file format.

    This class utilises `pytables` to read the DL1 file, and stores the
    information into the event containers.
    """
    _count = 0

    def __init__(self, config=None, tool=None, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

        try:
            import tables
        except ImportError:
            msg = "The `pytables` python module is required to access DL1DH data"
            self.log.error(msg)
            raise

        self.tables = tables

        self.metadata['is_simulation'] = True

    @staticmethod
    def is_compatible(file_path):
        import tables

        try:
            file = tables.File(file_path)
        except tables.HDF5ExtError:
            print("Not an HDF5 file")
            return False
        try:
            is_events = type(file.root.Events) == tables.table.Table
        except:
            print("Can't access events table")
            return False
        try:
            from packaging import version
            is_version = version.parse(file.root._v_attrs['dl1_data_handler_version']) > version.parse("0.7")
        except:
            print("Can't read dl1_data_handler version")
            return False

        return is_events & is_version

    def __exit__(self, exc_type, exc_val, exc_tb):
        DL1DHEventSource._count -= 1
        self.file.close()

    def _generator(self):
        with self.tables.open_file(self.input_url, mode='r') as self.file:
            # the container is initialized once, and data is replaced within
            # it after each yield
            counter = 0

            max_events = len(self.file.root.Events) if self.max_events is None else self.max_events
            eventstream = self.file.root.Events[:max_events]
            data = DataContainer()
            data.meta['origin'] = "dl1_data_handler"

            data.meta['input_url'] = self.input_url
            data.meta['max_events'] = self.max_events

            tel_types = set(self.file.root.Array_Information[:]['type'])

            tel_ids = {}
            for tel_type in tel_types:
                tel_ids[tel_type] = self.file.root.Array_Information \
                    [self.file.root.Array_Information[:]['type'] == tel_type]['id']

            # load subarray info
            data.inst.subarray = self._build_subarray_info()

            for event in eventstream:

                obs_id = event['obs_id']
                event_id = event['event_id']
                tels_with_data = set(np.concatenate([tel_ids[tel_type][event[tel_type.decode() + '_indices'].nonzero()]
                                                 for tel_type in tel_types]))
                data.count = counter
                data.r0.obs_id = obs_id
                data.r0.event_id = event_id
                data.r0.tels_with_data = tels_with_data
                data.r1.obs_id = obs_id
                data.r1.event_id = event_id
                data.r1.tels_with_data = tels_with_data
                data.dl0.obs_id = obs_id
                data.dl0.event_id = event_id
                data.dl0.tels_with_data = tels_with_data

                # handle telescope filtering by taking the intersection of
                # tels_with_data and allowed_tels
                if len(self.allowed_tels) > 0:
                    selected = tels_with_data & self.allowed_tels
                    if len(selected) == 0:
                        continue  # skip event
                    data.r0.tels_with_data = selected
                    data.r1.tels_with_data = selected
                    data.dl0.tels_with_data = selected

                # data.trig.tels_with_trigger = (self.file.
                #                                get_central_event_teltrg_list()) # info not kept

                # time_s, time_ns = self.file.get_central_event_gps_time()
                # data.trig.gps_time = Time(time_s * u.s, time_ns * u.ns,
                #                           format='unix', scale='utc')
                data.mc.energy = event['mc_energy'] * u.TeV
                data.mc.alt = Angle(event['alt'], u.rad)
                data.mc.az = Angle(event['az'], u.rad)
                data.mc.core_x = event['core_x'] * u.m
                data.mc.core_y = event['core_y'] * u.m
                data.mc.h_first_int = event['h_first_int'] * u.m
                data.mc.x_max = event['x_max'] * u.g / (u.cm**2)
                data.mc.shower_primary_id = int(event['shower_primary_id'])

                # mc run header data
                self._build_mcheader(data)

                # this should be done in a nicer way to not re-allocate the
                # data each time (right now it's just deleted and garbage
                # collected)

                data.r0.tel.clear()
                data.r1.tel.clear()
                data.dl0.tel.clear()
                data.dl1.tel.clear()
                data.mc.tel.clear()  # clear the previous telescopes

                for tel_type in tel_types:
                    idxs = event[tel_type.decode() + '_indices']
                    for idx in idxs[idxs > 0]:
                        tel_id = tel_ids[tel_type][np.where(idxs == idx)[0][0]]
                        charge = self.file.root[tel_type.decode()][idx]['charge']
                        peakpos = self.file.root[tel_type.decode()][idx]['peakpos']

                        data.dl1.tel[tel_id].image = charge[None, :]
                        data.dl1.tel[tel_id].peakpos = peakpos[None, :]

                yield data
                counter += 1

        return

    def _build_subarray_info(self):
        """
        constructs a SubarrayDescription object from the info in an DL1 file

        Parameters
        ----------
        file: pytables opened File

        Returns
        -------
        SubarrayDescription :
            instrumental information
        """

        subarray = SubarrayDescription("MonteCarloArray")

        for tel in self.file.root.Array_Information:
            tel_id = tel['id']
            tel_type = tel['type']
            subarray.tels[tel_id] = self._build_telescope_description(tel_type)
            tel_pos = u.Quantity([tel['x'], tel['y'], tel['z']], u.m)
            subarray.positions[tel_id] = tel_pos

        return subarray


    def _build_telescope_description(self, tel_type):

        tel_info = self.file.root.Telescope_Type_Information \
            [self.file.root.Telescope_Type_Information[:]['type'] == tel_type][0]

        camera_name = tel_info['camera'].decode()
        optics_name = tel_info['optics'].decode()
        try:
            CameraGeometry.from_name(camera_name)
        except ValueError:
            warnings.warn(f'Unkown camera name {camera_name}')
        try:
            OpticsDescription.from_name(optics_name)
        except ValueError:
            warnings.warn(f'Unkown optics name {optics_name}')

        return TelescopeDescription.from_name(optics_name, camera_name)


    def _build_mcheader(self, data):
        """
        Read the mcheader data from the DL1 file and update the data container

        Parameters
        ----------
        file: pytables opened file
        data: `ctapipe.io.containers.DataContainer`
        """

        for k in data.mcheader.keys():
            try:
                data.mcheader[k] = self.file.root._v_attrs[k]
            except:
                warnings.warn(f"item {k} does not exist in the file attributes")

        data.mcheader.run_array_direction = Angle(data.mcheader.run_array_direction, unit=u.rad)
        data.mcheader.energy_range_min *= u.TeV
        data.mcheader.energy_range_max *= u.TeV
        data.mcheader.prod_site_B_total *= u.uT
        data.mcheader.prod_site_B_declination *= u.rad
        data.mcheader.prod_site_B_inclination *= u.rad
        data.mcheader.prod_site_alt *= u.m
        data.mcheader.max_alt *= u.rad
        data.mcheader.min_alt *= u.rad
        data.mcheader.max_az *= u.rad
        data.mcheader.min_az *= u.rad
        data.mcheader.max_viewcone_radius *= u.deg
        data.mcheader.min_viewcone_radius *= u.deg
        data.mcheader.max_scatter_range *= u.m
        data.mcheader.min_scatter_range *= u.m
        data.mcheader.injection_height *= u.m
        data.mcheader.corsika_wlen_min *= u.nm
        data.mcheader.corsika_wlen_max *= u.nm