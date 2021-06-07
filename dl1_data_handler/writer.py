# -*- coding: utf-8 -*-
"""--Deprecated DL1 writer. Use ctapipe stage1 tool instead-- Load data from ctapipe EventSources and dump to file."""

from abc import ABC, abstractmethod
import pkg_resources
import os
import re
import multiprocessing
import logging
import math
import glob
import numpy as np
import tables
import uproot
from traitlets.config.loader import Config
import ctapipe
from ctapipe import io, calib
from ctapipe import containers
from ctapipe.image import cleaning, extractor, leakage, hillas_parameters, concentration, timing_parameters, morphology_parameters
from ctapipe.instrument.camera import CameraGeometry
from dl1_data_handler import table_definitions as table_defs
from dl1_data_handler import dl_eventsources
logger = logging.getLogger(__name__)


class DL1DataDumper(ABC):
    """Abstract class for dumping data from ctapipe DL1 containers to file."""

    @abstractmethod
    def __init__(self, output_filename):
        """Instantiate DL1DataDumper instance. Set event and image indices to initial values.
        Parameters
        ----------
        output_filename : str
            string filepath to output file.
        """
        self.output_filename = output_filename
        self.event_index = 0 # Be sure to initialize self.event_index

    # Write a single event's information (dl1 data, monte carlo information)
    @abstractmethod
    def dump_event(self, event_container):
        """Dump ctapipe event data (event params and images) to output file.
        Parameters
        ----------
        event_container : ctapipe.io.containers.DataContainer
            ctapipe parent event container.
        """
        self.event_index += 1 # Be sure to increment self.event_index

    # Write a single event's information (dl1 data, monte carlo information)
    @abstractmethod
    def dump_mc_event(self, eventio_mc_event, obs_id):
        """Dump mc event data (event params and images) to output file.
        Parameters
        ----------
        eventio_mc_event : dict
            dictionary yielded by eventio.simtel.SimTelFile.iter_mc_events()
        obs_id : int
            observation/run id
        """
        pass

    # Prepare the file's header, telescope/array descriptions, event and image tables
    @abstractmethod
    def prepare_file(self, input_filename, subarray, mcheader, cleaning_algorithm_metadata):
        """Dump file-level data to file and setup file structure.
        Creates Event and image tables. Sets self.subarray for later fast lookup.
        Parameters
        ----------
        input_filename : str
            filename of input file being written
        subarray : ctapipe.io.instrument.SubarrayDescription
            ctapipe subarray description object.
        mcheader : ctapipe.io.containers.MCHeaderContainer
            ctapipe container of monte carlo header data (for entire run).
        cleaning_algorithm_metadata: dict
            cleaning algortithm name and args stored in datawriter part of config file
        """
        pass


class CTAMLDataDumper(DL1DataDumper):
    """Class for dumping ctapipe DL1 data to the CTA ML data format.
    See the Github repository wiki page for a detailed description of the data
    format.
    Attributes
    ----------
    DEFAULT_IMGS_PER_EVENT : float
        Default number of triggered telescopes (images) expected for all
        telescopes. This value is used as a default if a given telescope type's
        expected_images_per_event is not specified.
    """

    DEFAULT_IMGS_PER_EVENT = 1.0

    def __init__(self,
                 output_filename,
                 filter_settings=None,
                 expected_tel_types=10,
                 expected_tels=300,
                 expected_events=100,
                 expected_mc_events=50000,
                 expected_images_per_event=None,
                 index_columns=None,
                 save_mc_events=False,
                 cleaning_settings=None):
        """Instantiate a CTAMLDataDumper instance.
        Parameters
        ----------
        output_filename : str
            String path to output file.
        filter_settings : dict
            Dictionary of filter settings (kwargs), passed to the constructor
            for tables.Filters. Determines compression settings.
        expected_tel_types : int
            Number of expected telescope types in the
            '/Telescope_Type_Information' table. Used for setting the chunk
            size.
        expected_tels : int
            Number of expected telescope types in the
            '/Telescope_Type_Information' table. Used for setting the chunk
            size.
        expected_events : int
            Number of expected telescope types in the
            '/Telescope_Type_Information' table. Used for setting the chunk
            size.
        expected_images_per_event : dict
            Dictionary containing telescope type names as keys, with the
            expected average number of triggered telescopes of that type per
            event (float) as the value. Used for setting the chunk size.
        index_columns : list
            List of tuples of form (table_path, column_name), specifying the
            tables and columns in the output file on which to create indexes
            for faster search. Used for setting the chunk size.
        """
        super().__init__(output_filename)
        self.file = tables.open_file(output_filename, mode="w")

        if filter_settings is None:
            self.filter_settings = {
                     'complib': 'lzo',
                     'complevel': 1
            }
        else:
            self.filter_settings = filter_settings
        self.filters = tables.Filters(**self.filter_settings)

        self.expected_tel_types = expected_tel_types
        self.expected_tels = expected_tels
        self.expected_events = expected_events
        self.expected_mc_events = expected_mc_events

        if expected_images_per_event is None:
            self.expected_images_per_event = {
                     'LSTCam': 0.5,
                     'NectarCam': 2.0,
                     'FlashCam': 2.0,
                     'SCTCam': 1.5,
                     'DigiCam': 1.25,
                     'ASTRICam': 1.25,
                     'CHEC': 1.25,
                     'MAGICCam': 2.0,
                 },
        else:
            self.expected_images_per_event = expected_images_per_event

        if index_columns is None:
            self.index_columns = [
                     ('/Events', 'mc_energy'),
                     ('/Events', 'alt'),
                     ('/Events', 'az'),
                     ('tel', 'event_index')
                 ]
        else:
            self.index_columns = index_columns

        self.tel_tables = []
        self.subarray = None
        self.cam_geometry = None
        self.event_index = 0
        self.image_indices = {}

        self.save_mc_events = save_mc_events

        self.cleaning_settings = (cleaning_settings
                                  if cleaning_settings else {})

    def __del__(self):
        """Cleanup + finalize output file."""
        # Flush all tables
        self.file.flush()

        self.finalize()
        try:
            self.file.close()
        except:
            pass

    def dump_instrument_info(self, subarray):
        """Dump ctapipe instrument container to output file.
        If not present in the output file, creates two tables,
        '/Array_Information' and '/Telescope_Type_Information'. Then,
        populates them row by row with array data and telescope type data.
        Parameters
        ----------
        subarray : ctapipe.io.instrument.SubarrayDescription
            ctapipe subarray description object.
        """

        if "/Array_Information" in self.file:
            array_table = self.file.root.Array_Information
            logger.info("Array_Information table already present. Validating...")
            for tel_id in subarray.tels:
                tel_desc = subarray.tels[tel_id]

                if str(tel_desc) != "LST_MAGIC_MAGICCam":
                    rows = [row for row in array_table.iterrows()
                            if row["id"] == tel_id and
                            row["type"].decode('utf-8') == str(tel_desc) and
                            row["x"] == subarray.positions[tel_id].value[0] and
                            row["y"] == subarray.positions[tel_id].value[1] and
                            row["z"] == subarray.positions[tel_id].value[2]]

                    if len(rows) != 1:
                        logger.error("Printing all entries in Array_Information...")
                        for row in array_table.iterrows():
                            logger.error("{}, {}, [{}, {}, {}]".format(row["id"], row["type"].decode('utf-8'), row["x"], row["y"], row["z"]))
                        logger.error("Failed to find: {}, {}, {}".format(tel_id, str(tel_desc), subarray.positions[tel_id].value))
                        raise ValueError("Failed to validate telescope description in Array_Information.")

        else:
            array_table = self._create_array_table()
            row = array_table.row

            logger.info("Writing array/subarray information to table...")
            for tel_id in subarray.tels:
                tel_desc = subarray.tels[tel_id]
                row["id"] = tel_id
                row["type"] = str(tel_desc)
                row["x"] = subarray.positions[tel_id].value[0]
                row["y"] = subarray.positions[tel_id].value[1]
                row["z"] = subarray.positions[tel_id].value[2]
                row.append()
            array_table.flush()

        if "/Telescope_Type_Information" in self.file:
            tel_table = self.file.root.Telescope_Type_Information
            max_px = max([len(x.camera.geometry.pix_id) for x in subarray.tel.values()])

            logger.info("Telescope_Type_Information table already present. Validating...")
            for tel_type in subarray.telescope_types:
                tel_id = subarray.get_tel_ids_for_type(tel_type)[0]
                tel_desc = subarray.tels[tel_id]

                pos = np.zeros(shape=(max_px, 2))
                x_len = subarray.tel[tel_id].camera.geometry.pix_x.value.shape[0]
                y_len = subarray.tel[tel_id].camera.geometry.pix_y.value.shape[0]
                pos[0:x_len, 0] = subarray.tel[tel_id].camera.geometry.pix_x.value
                pos[0:y_len, 1] = subarray.tel[tel_id].camera.geometry.pix_y.value
                pix_rotation = subarray.tel[tel_id].camera.geometry.pix_rotation.value
                cam_rotation = subarray.tel[tel_id].camera.geometry.cam_rotation.value

                rows = [row for row in tel_table.iterrows() if
                            row["type"].decode('utf-8') == str(tel_desc) and
                            row["optics"].decode('utf-8') == str(tel_desc.optics) and
                            row["camera"].decode('utf-8') == str(tel_desc.camera) and
                            row["num_pixels"] == len(subarray.tel[tel_id].camera.geometry.pix_id) and
                            np.allclose(row["pixel_positions"], pos) and
                            np.around(row["pix_rotation"], decimals=1) == np.around(pix_rotation,1) and
                            row["cam_rotation"] == cam_rotation]

                if len(rows) != 1:
                    for row in tel_table.iterrows():
                        logger.error("{}, {}, {}, {}, {}, {}".format(row["type"].decode('utf-8'), row["optics"].decode('utf-8'), row["camera"].decode('utf-8'), row["num_pixels"], row["pix_rotation"], row["cam_rotation"]))
                        logger.error(row["pixel_positions"])
                    logger.error("New input file: {}-{}-{}-{}-{}-{}".format(str(tel_desc), str(tel_desc.optics), str(tel_desc.camera), len(subarray.tel[tel_id].camera.geometry.pix_id), pix_rotation, cam_rotation))
                    logger.error(pos)
                    raise ValueError("Failed to validate telescope type description in Telescope_Type_Information.")
        else:
            # Compute maximum number of pixels across all camera types
            max_px = max([len(x.camera.geometry.pix_id) for x in subarray.tel.values()])
            tel_table = self._create_tel_table(subarray, max_px)
            row = tel_table.row

            logger.info("Writing telescope type information to table...")
            for tel_type in subarray.telescope_types:
                tel_id = subarray.get_tel_ids_for_type(tel_type)[0]
                tel_description = subarray.tels[tel_id]

                pos = np.zeros(shape=(max_px, 2))
                x_len = subarray.tel[tel_id].camera.geometry.pix_x.value.shape[0]
                y_len = subarray.tel[tel_id].camera.geometry.pix_y.value.shape[0]
                pos[0:x_len, 0] = subarray.tel[tel_id].camera.geometry.pix_x.value
                pos[0:y_len, 1] = subarray.tel[tel_id].camera.geometry.pix_y.value

                row["type"] = str(tel_description)
                row["optics"] = str(tel_description.optics)
                row["camera"] = str(tel_description.camera)
                row["num_pixels"] = len(subarray.tel[tel_id].camera.geometry.pix_id)
                row["pixel_positions"] = pos
                row["pix_rotation"] = np.around(subarray.tel[tel_id].camera.geometry.pix_rotation.value, decimals=1)
                row["cam_rotation"] = subarray.tel[tel_id].camera.geometry.cam_rotation.value
                row.append()
            tel_table.flush()

    def dump_mc_header_info(self, mcheader_container, tel_desc):
        """Dump ctapipe instrument container to output file.
        Dumps entire contents of MC header container without selection.
        Parameters
        ----------
        mc_header_container : ctapipe.io.containers.MCHeaderContainer
            ctapipe container of monte carlo header data (for entire run).
        tel_desc : str
            telescope type to skip check for MAGIC files
        """
        logger.info("Writing MC header information to file attributes...")

        attributes = self.file.root._v_attrs
        mcheader_dict = mcheader_container.as_dict()

        for field in mcheader_dict:
            if field in attributes:
                if field == "num_showers":
                    attributes[field] = attributes[field] + mcheader_dict[field]
                elif field == "run_array_direction":
                    if not np.allclose(attributes[field], mcheader_dict[field].value):
                        raise ValueError("Attribute {} in output file root attributes does not match new value in input file: {} vs {}".format(field, attributes[field], mcheader_dict[field].value))
                elif field == "shower_prog_start" or field == "detector_prog_start":
                    continue
                else:
                    if str(tel_desc) != "LST_MAGIC_MAGICCam":
                        if hasattr(mcheader_dict[field], 'value'):
                            match = math.isclose(attributes[field], mcheader_dict[field].value)
                        elif type(mcheader_dict[field]) is str or type(mcheader_dict[field]) is int:
                            match = (attributes[field] == mcheader_dict[field])
                        elif type(mcheader_dict[field]) is float:
                            match = math.isclose(attributes[field], mcheader_dict[field])
                        else:
                            raise ValueError("Found unexpected type for field {} in MC header: {}".format(field, type(mcheader_dict[field])))

                        if not match:
                            raise ValueError("Attribute {} in output file root attributes does not match new value in input file: {} vs {}".format(field, attributes[field], mcheader_dict[field]))
            else:
                attributes[field] = mcheader_dict[field]

    def dump_header_info(self, input_filename):
        """Dump all non-ctapipe header data to output file.
        Uses pkg_resources to get software versions in current Python
        installation.
        Parameters
        ----------
        input_filename : str
            Full path to input file being dumped.
        """
        logger.info(
            "Writing general header information to file attributes...")

        attributes = self.file.root._v_attrs

        if not hasattr(attributes, 'dl1_data_handler_version'):
            attributes['dl1_data_handler_version'] = pkg_resources.get_distribution('dl1_data_handler').version
        if not hasattr(attributes, 'ctapipe_version'):
            attributes['ctapipe_version'] = pkg_resources.get_distribution('ctapipe').version

        if not hasattr(attributes, 'runlist'):
            attributes.runlist = []
        attributes.runlist = attributes.runlist + [os.path.basename(
            input_filename)]

    def dump_event(self, event_container):
        """Dump ctapipe event data (event params and images) to output file.
        Creates '/Events' table in output file if not present, then does the
        same for all required image tables. Finally, writes all event
        parameters and images to tables.
        Parameters
        ----------
        event_container : ctapipe.io.containers.DataContainer
            ctapipe container of all event data for a given event.
        """
        event_row = self.file.root.Events.row
        event_row['event_id'] = event_container.index.event_id
        event_row['obs_id'] = event_container.index.obs_id

        if event_container.mc:
            event_row['shower_primary_id'] = (
                event_container.mc.shower_primary_id)
            event_row['core_x'] = event_container.mc.core_x.value
            event_row['core_y'] = event_container.mc.core_y.value
            event_row['h_first_int'] = event_container.mc.h_first_int.value
            event_row['x_max'] = event_container.mc.x_max.value
            event_row['mc_energy'] = event_container.mc.energy.value
            event_row['log_mc_energy'] = np.log10(event_container.mc.energy.value)
            event_row['alt'] = event_container.mc.alt.value
            event_row['az'] = event_container.mc.az.value
            event_row['array_pointing_alt'] = event_container.pointing.array_altitude.value
            event_row['array_pointing_az'] = event_container.pointing.array_azimuth.value
            # North pointing correction
            delta_alt = event_row['alt']-event_row['array_pointing_alt']
            delta_az = event_row['az']-event_row['array_pointing_az']
            if delta_az > np.pi:
                delta_az -= 2*np.pi
            elif delta_az < -np.pi:
                delta_az += 2*np.pi
            event_row['delta_direction'] = np.array([delta_alt, delta_az], np.float32)

        store_event = False
        for tel_type in self.subarray:
            image_table = self.file.get_node(
                '/Images/' + str(tel_type),
                classname='Table')
            image_row = image_table.row

            index_vector = []
            for tel_id in self.subarray[tel_type]:
                if tel_id in event_container.dl1.tel:
                    index_vector.append(self.image_indices[tel_type])
                    self.image_indices[tel_type] += 1
                else:
                    index_vector.append(0)
            if not np.any(index_vector):
                continue
            store_event = True

            for tel_id in self.subarray[tel_type]:
                if tel_id in event_container.dl1.tel:
                    image_row['charge'] = event_container.dl1.tel[tel_id].image
                    image_row['peak_time'] = event_container.dl1.tel[tel_id].peak_time
                    image_row["event_index"] = self.event_index

                    for index_parameters_table in range(0, len(self.cleaning_settings)+1):
                        parameter_table = self.file.get_node(
                            '/Parameters' + str(index_parameters_table) + '/' + str(tel_type),
                            classname='Table')
                        parameter_row = parameter_table.row

                        parameter_row["event_index"] = self.event_index

                        if index_parameters_table == 0:
                            image_row['image_mask0'] = event_container.dl1.tel[tel_id].image_mask
                            image_row['inv_image_mask0'] = np.invert(event_container.dl1.tel[tel_id].image_mask)

                            parameter_row["event_index"] = self.event_index
                            parameter_row["leakage_intensity_1"] = event_container.dl1.tel[
                                tel_id].parameters.leakage.intensity_width_1
                            parameter_row["leakage_intensity_2"] = event_container.dl1.tel[
                                tel_id].parameters.leakage.intensity_width_2
                            parameter_row["leakage_pixels_1"] = event_container.dl1.tel[
                                tel_id].parameters.leakage.pixels_width_1
                            parameter_row["leakage_pixels_2"] = event_container.dl1.tel[
                                tel_id].parameters.leakage.pixels_width_2

                            parameter_row["hillas_intensity"] = event_container.dl1.tel[
                                tel_id].parameters.hillas.intensity
                            parameter_row["hillas_log_intensity"] = np.log10(event_container.dl1.tel[
                                tel_id].parameters.hillas.intensity)
                            parameter_row["hillas_x"] = event_container.dl1.tel[tel_id].parameters.hillas.x.value
                            parameter_row["hillas_y"] = event_container.dl1.tel[tel_id].parameters.hillas.y.value
                            parameter_row["hillas_r"] = event_container.dl1.tel[tel_id].parameters.hillas.r.value
                            parameter_row["hillas_phi"] = event_container.dl1.tel[
                                tel_id].parameters.hillas.phi.value
                            parameter_row["hillas_length"] = event_container.dl1.tel[
                                tel_id].parameters.hillas.length.value
                            parameter_row["hillas_width"] = event_container.dl1.tel[
                                tel_id].parameters.hillas.width.value
                            parameter_row["hillas_psi"] = event_container.dl1.tel[
                                tel_id].parameters.hillas.psi.value
                            parameter_row["hillas_skewness"] = event_container.dl1.tel[
                                tel_id].parameters.hillas.skewness
                            parameter_row["hillas_kurtosis"] = event_container.dl1.tel[
                                tel_id].parameters.hillas.kurtosis

                            parameter_row['concentration_cog'] = event_container.dl1.tel[
                                tel_id].parameters.concentration.cog
                            parameter_row['concentration_core'] = event_container.dl1.tel[
                                tel_id].parameters.concentration.core
                            parameter_row['concentration_pixel'] = event_container.dl1.tel[
                                tel_id].parameters.concentration.pixel

                            parameter_row['timing_slope'] = event_container.dl1.tel[tel_id].parameters.timing.slope.value
                            parameter_row['timing_slope_err'] = event_container.dl1.tel[tel_id].parameters.timing.slope_err.value
                            parameter_row['timing_intercept'] = event_container.dl1.tel[tel_id].parameters.timing.intercept
                            parameter_row['timing_intercept_err'] = event_container.dl1.tel[tel_id].parameters.timing.intercept_err
                            parameter_row['timing_deviation'] = event_container.dl1.tel[tel_id].parameters.timing.deviation

                            parameter_row['morphology_num_pixels'] = event_container.dl1.tel[
                                tel_id].parameters.morphology.num_pixels
                            parameter_row['morphology_num_islands'] = event_container.dl1.tel[
                                tel_id].parameters.morphology.num_islands
                            parameter_row['morphology_num_small_islands'] = event_container.dl1.tel[
                                tel_id].parameters.morphology.num_small_islands
                            parameter_row['morphology_num_medium_islands'] = event_container.dl1.tel[
                                tel_id].parameters.morphology.num_medium_islands
                            parameter_row['morphology_num_large_islands'] = event_container.dl1.tel[
                                tel_id].parameters.morphology.num_large_islands

                        else:

                            cleaning_method = getattr(cleaning,
                                                      self.cleaning_settings[index_parameters_table - 1]['algorithm'])

                            cleanmask = cleaning_method(self.cam_geometry[tel_type],
                                                        event_container.dl1.tel[tel_id].image,
                                                        **self.cleaning_settings[index_parameters_table - 1]['args'])

                            image_row["image_mask"+str(index_parameters_table)] = cleanmask
                            image_row["inv_image_mask"+str(index_parameters_table)] = np.invert(cleanmask)

                            leakage_values = containers.LeakageContainer()
                            hillas_parameters_values = containers.HillasParametersContainer()
                            concentration_values = containers.ConcentrationContainer()
                            timing_values = containers.TimingParametersContainer()
                            morphology_values = containers.MorphologyContainer()

                            if any(cleanmask):
                                leakage_values = leakage(self.cam_geometry[tel_type],
                                                         event_container.dl1.tel[tel_id].image,
                                                         cleanmask)

                                hillas_parameters_values = hillas_parameters(
                                    self.cam_geometry[tel_type][cleanmask],
                                    event_container.dl1.tel[tel_id].image[cleanmask])

                                concentration_values = concentration(self.cam_geometry[tel_type],
                                                                     event_container.dl1.tel[tel_id].image,
                                                                     hillas_parameters_values)
                                try:
                                    timing_values = timing_parameters(self.cam_geometry[tel_type],
                                                                  event_container.dl1.tel[tel_id].image,
                                                                  event_container.dl1.tel[tel_id].peak_time,
                                                                  hillas_parameters_values,
                                                                  cleanmask)
                                except:
                                    timing_values = containers.TimingParametersContainer()

                                morphology_values = morphology_parameters(
                                    self.cam_geometry[tel_type],
                                    cleanmask)

                            # leakage
                            parameter_row["leakage_intensity_1"] = leakage_values['intensity_width_1']
                            parameter_row["leakage_intensity_2"] = leakage_values['intensity_width_2']
                            parameter_row["leakage_pixels_1"] = leakage_values['pixels_width_1']
                            parameter_row["leakage_pixels_2"] = leakage_values['pixels_width_2']

                            # hillas
                            parameter_row["hillas_intensity"] = hillas_parameters_values['intensity']
                            parameter_row["hillas_log_intensity"] = np.log10(hillas_parameters_values['intensity'])
                            parameter_row["hillas_x"] = hillas_parameters_values['x'].value
                            parameter_row["hillas_y"] = hillas_parameters_values['y'].value
                            parameter_row["hillas_r"] = hillas_parameters_values['r'].value
                            parameter_row["hillas_phi"] = hillas_parameters_values['phi'].value
                            parameter_row["hillas_length"] = hillas_parameters_values['length'].value
                            parameter_row["hillas_width"] = hillas_parameters_values['width'].value
                            parameter_row["hillas_psi"] = hillas_parameters_values['psi'].value
                            parameter_row["hillas_skewness"] = hillas_parameters_values['skewness']
                            parameter_row["hillas_kurtosis"] = hillas_parameters_values['kurtosis']

                            # concentration
                            parameter_row['concentration_cog'] = concentration_values['cog']
                            parameter_row['concentration_core'] = concentration_values['core']
                            parameter_row['concentration_pixel'] = concentration_values['pixel']

                            # timing
                            parameter_row['timing_deviation'] = timing_values['deviation']
                            parameter_row['timing_intercept'] = timing_values['intercept']
                            parameter_row['timing_intercept_err'] = timing_values['intercept_err']
                            parameter_row['timing_slope'] = timing_values['slope'].value
                            parameter_row['timing_slope_err'] = timing_values['slope_err'].value

                            # morphology
                            parameter_row['morphology_num_pixels'] = morphology_values['num_pixels']
                            parameter_row['morphology_num_islands'] = morphology_values['num_islands']
                            parameter_row['morphology_num_small_islands'] = morphology_values[
                                'num_small_islands']
                            parameter_row['morphology_num_medium_islands'] = morphology_values['num_medium_islands']
                            parameter_row['morphology_num_large_islands'] = morphology_values[
                                'num_large_islands']

                        parameter_row.append()

                    image_row.append()

            event_row[str(tel_type) + '_indices'] = index_vector
            event_row[str(tel_type) + '_multiplicity'] = sum(
                index > 0 for index in index_vector)

        if store_event:
            event_row.append()
            self.event_index += 1

    def dump_mc_event(self, eventio_mc_event, obs_id):
        """Dump eventio event data (event params and images) to output file.
        Creates '/Events' table in output file if not present, then does the
        same for all required image tables. Finally, writes all event
        parameters and images to tables.
        Parameters
        ----------
        eventio_mc_event : dict
            eventio mc event dictionary (yielded from eventio.SimTelFile.iter_mc_events())
        obs_id : int
            run/observation number for the event
        """
        event_row = self.file.root.MC_Events.row

        event_row['event_id'] = eventio_mc_event['event_id']
        event_row['obs_id'] = obs_id
        event_row['shower_primary_id'] = eventio_mc_event['mc_shower']['primary_id']
        event_row['core_x'] = eventio_mc_event['mc_event']['xcore']
        event_row['core_y'] = eventio_mc_event['mc_event']['ycore']
        event_row['h_first_int'] = eventio_mc_event['mc_shower']['h_first_int']
        event_row['x_max'] = eventio_mc_event['mc_shower']['xmax']
        event_row['mc_energy'] = eventio_mc_event['mc_shower']['energy']
        event_row['log_mc_energy'] = eventio_mc_event['mc_shower']['log_mc_energy']
        event_row['alt'] = eventio_mc_event['mc_shower']['altitude']
        event_row['az'] = eventio_mc_event['mc_shower']['azimuth']
        event_row['array_pointing_alt'] = eventio_mc_event['mc_shower']['array_pointing_alt']
        event_row['array_pointing_az'] = eventio_mc_event['mc_shower']['array_pointing_az']
        event_row['delta_direction'] = event_row['delta_direction']

        event_row.append()

    def _create_event_table(self, subarray):
        # Create event table
        event_table_desc = table_defs.EventTableRow

        if self.save_mc_events:
            mc_event_table = self.file.create_table(self.file.root,
                                                    'MC_Events',
                                                    event_table_desc,
                                                    "Table of MC Event Information",
                                                    filters=self.filters,
                                                    expectedrows=(
                                                     self.expected_mc_events))

        for tel_type in subarray.telescope_types:
            event_table_desc.columns[
                str(tel_type) + '_indices'] = (
                tables.UInt32Col(shape=(
                    len(subarray.get_tel_ids_for_type(tel_type)))))
            event_table_desc.columns[
                str(tel_type) + '_multiplicity'] = (
                tables.UInt32Col())

        event_table = self.file.create_table(self.file.root,
                                             'Events',
                                             event_table_desc,
                                             "Table of Event Information",
                                             filters=self.filters,
                                             expectedrows=(
                                                 self.expected_events))

    def _create_image_tables(self, subarray):
        self.file.create_group(self.file.root, "Images")
        for tel_desc in set(subarray.tels.values()):
            tel_name = str(tel_desc)

            if ("/{}".format(tel_name)) not in self.file.root.Images:
                logger.info("Creating {} image table...".format(tel_name))
                self.tel_tables.append(tel_name)

                image_shape = (len(tel_desc.camera.geometry.pix_id),)

                columns_dict = {
                    "event_index": tables.Int32Col(),
                    "charge": tables.Float32Col(shape=image_shape),
                    "peak_time": tables.Float32Col(shape=image_shape),
                }

                for index_parameters_table in range(0, len(self.cleaning_settings)+1):
                    columns_dict["image_mask"+str(index_parameters_table)] = tables.BoolCol(shape=image_shape)
                    columns_dict["inv_image_mask"+str(index_parameters_table)] = tables.BoolCol(shape=image_shape)

                description = type('description',
                                  (tables.IsDescription,),
                                  columns_dict)

                # Calculate expected number of rows for compression
                if tel_name in self.expected_images_per_event:
                    expected_rows = (
                        self.expected_events * self.expected_images_per_event[
                                tel_name])
                else:
                    expected_rows = (
                        self.DEFAULT_IMGS_PER_EVENT * self.expected_events)

                image_table = self.file.create_table(
                         self.file.root.Images,
                         tel_name,
                         description,
                         "Image table of {} images".format(tel_name),
                         filters=self.filters,
                         expectedrows=expected_rows)



                # Place blank image at index 0 of all image tables
                image_row = image_table.row

                image_row['charge'] = np.zeros(image_shape, dtype=np.float32)
                image_row['event_index'] = -1
                image_row['peak_time'] = np.zeros(image_shape, dtype=np.float32)
                for index_parameters_table in range(0, len(self.cleaning_settings)+1):
                    image_row["image_mask"+str(index_parameters_table)] = np.zeros(image_shape, dtype=np.bool_)
                    image_row["inv_image_mask"+str(index_parameters_table)] = np.ones(image_shape, dtype=np.bool_)
                image_row.append()
                image_table.flush()

    def _create_parameter_tables(self, subarray, index_parameters_table, cleaning_main_algorithm_metadata):
        self.file.create_group(self.file.root, "Parameters"+str(index_parameters_table))
        for tel_desc in set(subarray.tels.values()):
            tel_name = str(tel_desc)
            parameter_table_number = getattr(self.file.root, "Parameters" + str(index_parameters_table))
            if ("/{}".format(tel_name)) not in parameter_table_number:
                logger.info("Creating {} parameter table...".format(tel_name))

                # Calculate expected number of rows for compression
                if tel_name in self.expected_images_per_event:
                    expected_rows = (
                        self.expected_events * self.expected_images_per_event[
                            tel_name])
                else:
                    expected_rows = (
                        self.DEFAULT_IMGS_PER_EVENT * self.expected_events)

                if index_parameters_table == 0:
                    parameter_table = self.file.create_table(parameter_table_number,
                                                         tel_name,
                                                         table_defs.ParametersTableRow,
                                                         "Parameter table of {} parameters, algorithm: {}, args: {}".format(tel_name,
                                                                                           cleaning_main_algorithm_metadata['algorithm'],
                                                                                           cleaning_main_algorithm_metadata['args']),
                                                         filters=self.filters,
                                                         expectedrows=expected_rows)
                else :
                    parameter_table = self.file.create_table(parameter_table_number,
                                                         tel_name,
                                                         table_defs.ParametersTableRow,
                                                         "Parameter table of {} parameters, algorithm: {}, args: {}".format(tel_name,
                                                                  self.cleaning_settings[index_parameters_table -1]['algorithm'],
                                                                  self.cleaning_settings[index_parameters_table -1]['args']),
                                                        filters=self.filters,
                                                        expectedrows=expected_rows)

                # Place blank image at index 0 of all image tables
                parameter_row = parameter_table.row

                parameter_row['event_index'] = -1

                parameter_row['leakage_intensity_1'] = np.float32(np.nan)
                parameter_row['leakage_intensity_2'] = np.float32(np.nan)
                parameter_row['leakage_pixels_1'] = np.float32(np.nan)
                parameter_row['leakage_pixels_2'] = np.float32(np.nan)

                parameter_row['hillas_intensity'] = np.float32(np.nan)
                parameter_row['hillas_log_intensity'] = np.float32(np.nan)
                parameter_row['hillas_x'] = np.float32(np.nan)
                parameter_row['hillas_y'] = np.float32(np.nan)
                parameter_row['hillas_r'] = np.float32(np.nan)
                parameter_row['hillas_phi'] = np.float32(np.nan)
                parameter_row['hillas_length'] = np.float32(np.nan)
                parameter_row['hillas_width'] = np.float32(np.nan)
                parameter_row['hillas_psi'] = np.float32(np.nan)
                parameter_row['hillas_skewness'] = np.float32(np.nan)
                parameter_row['hillas_kurtosis'] = np.float32(np.nan)

                parameter_row['concentration_cog'] = np.float32(np.nan)
                parameter_row['concentration_core'] = np.float32(np.nan)
                parameter_row['concentration_pixel'] = np.float32(np.nan)

                parameter_row['timing_slope'] = np.float32(np.nan)
                parameter_row['timing_slope_err'] = np.float32(np.nan)
                parameter_row['timing_intercept'] = np.float32(np.nan)
                parameter_row['timing_intercept_err'] = np.float32(np.nan)
                parameter_row['timing_deviation'] = np.float32(np.nan)

                parameter_row['morphology_num_pixels'] = -1
                parameter_row['morphology_num_islands'] = -1
                parameter_row['morphology_num_small_islands'] = -1
                parameter_row['morphology_num_medium_islands'] = -1
                parameter_row['morphology_num_large_islands'] = -1

                parameter_row.append()
                parameter_table.flush()

    def prepare_file(self, input_filename, subarray, mcheader, cleaning_algorithm_metadata):
        """Dump file-level data to file and setup file structure.
        Creates Event and image tables. Sets self.subarray for later fast lookup.
        Parameters
        ----------
        input_filename : str
            filename of input file being written
        subarray : ctapipe.io.instrument.SubarrayDescription
            ctapipe subarray description object.
        mcheader : ctapipe.io.containers.MCHeaderContainer
            ctapipe container of monte carlo header data (for entire run).
        cleaning_algorithm_metadata: dict
            cleaning algortithm name and args stored in datawriter part of config file
        """
        try:
            self.dump_header_info(input_filename)
            self.dump_instrument_info(subarray)
            self.dump_mc_header_info(mcheader, subarray.tels[1])
            if "/Events" not in self.file:
                self._create_event_table(subarray)
            if "/Images" not in self.file:
                self._create_image_tables(subarray)
            if "/Parameters0" not in self.file:
                for index_parameters_table in range(0, len(self.cleaning_settings)+1):
                    self._create_parameter_tables(subarray, index_parameters_table, cleaning_algorithm_metadata)

            if self.subarray:
                for tel_type in self.subarray:
                    if sorted(subarray.get_tel_ids_for_type(tel_type)) != self.subarray[tel_type]:
                        raise ValueError("Tel ids in new input file {} do not match the"""
                                         " description in the current output file.".format(input_filename))
            else:
                self.subarray = {tel_type: sorted(subarray.get_tel_ids_for_type(tel_type))
                                 for tel_type in subarray.telescope_types}
                self.cam_geometry = {tel_type: tel_type.camera.geometry
                                 for tel_type in subarray.telescope_types}

            for tel_type in self.subarray:
                if tel_type not in self.image_indices:
                    self.image_indices[tel_type] = 1

        except IOError:
            logger.error("Failed to write header info from file "
                         "{}".format(
                             input_filename))
            raise IOError

    def _create_array_table(self):
        logger.info("Creating array info table...")
        array_table = self.file.create_table(self.file.root,
                                             'Array_Information',
                                             table_defs.ArrayTableRow,
                                             ("Table of array/subarray "
                                              "information"),
                                             filters=self.filters,
                                             expectedrows=(
                                                 self.expected_tels)
                                             )
        return array_table

    def _create_tel_table(self, subarray, max_px):
        # Create a row description object for the telescope table
        tel_table_desc = table_defs.TelTableRow

        # Add a column field for the pixel position map
        tel_table_desc.columns['pixel_positions'] = tables.Float32Col(
            shape=(max_px, 2))

        # Create telescope information table
        tel_table = self.file.create_table(
            self.file.root,
            'Telescope_Type_Information',
            tel_table_desc,
            "Table of telescope type information",
            filters=self.filters,
            expectedrows=self.expected_tel_types)

        return tel_table

    def finalize(self):
        """Do final processing before closing file.
        Currently only adds indexes to requested columns.
        """
        # Add all requested PyTables column indexes to tables
        if self.index_columns:
            logger.info("Adding indexed columns...")
            for location, col_name in self.index_columns:
                if location == 'tel':
                    table_names = ["/" + i for i in self.tel_tables]
                else:
                    table_names = [location]

                for table_name in table_names:
                    if ("/Images/{}".format(table_name)) in self.file:
                        table_name = "/Images/{}".format(table_name)
                    for index_parameters_table in range(0, len(self.cleaning_settings)+1):
                        if ("/Parameters"+str(index_parameters_table)+"/{}".format(table_name)) in self.file:
                            table_name = "/Parameters"+str(index_parameters_table)+"/{}".format(table_name)
                    try:
                        table = self.file.get_node(table_name,
                                                   classname='Table')
                        table.cols._f_col(col_name).create_index()
                        logger.info("Added index on {}:{}".format(
                            table_name,
                            col_name))
                    except Exception:
                        logger.warning(
                            "Failed to create index on {} : {}".format(
                                table_name,
                                col_name))
                        pass


class DL1DataWriter:
    """Writes data using event sources and DL1DataDumpers.
    Provides some options for controlling the output file sizes.
    """

    def __init__(self,
                 event_source_class=None,
                 event_source_settings=None,
                 selected_telescope_ids=None,
                 data_dumper_class=CTAMLDataDumper,
                 data_dumper_settings=None,
                 preselection_cut_function=None,
                 write_mode='parallel',
                 output_file_size=10737418240,
                 events_per_file=None,
                 save_mc_events=False,
                 cleaning_settings=None,
                 gain_selector_settings=None,
                 image_extractor_settings=None):
        """Initialize a DL1DataWriter instance.
        Provides some options for controlling the output file sizes.
        Parameters
        ----------
        event_source_class : subclass of ctapipe.io.eventsource.EventSource
            A subclass of EventSource which will be used to load and yield
            events as DataContainers.
        event_source_settings : dict
            A dictionary of kwargs which will be passed into the constructor
            for the EventSource.
        selected_telescope_ids : set of telescope, which should be included in the
            subarray
        data_dumper_class : subclass of dl1_data_writer.DL1DataDumper
            A subclass of DL1DataDumper which will be used to write events from
            the EventSource to output files.
        data_dumper_settings : dict
            A dictionary of kwargs which will be passed into the constructor
            for the DL1DataDumper.
        preselection_cut_function : function
            A cut function used to determine which events in the input files
            to write to the output files. Takes a
            ctapipe.io.containers.DataContainer describing a single event and
            returns a boolean indicating if it passes the cut. If None, no cut
            will be applied.
        write_mode : str
            Whether to process the data with parallel threads (one per run)
            or in serial. Valid options are 'serial' and 'parallel'.
        output_file_size : int
            Maximum size of each output file. If the total amount of input data
            requested for a given output file exceeds this size, the output
            will be split across multiple files.
        events_per_file : int
            Maximum number of events to write per output file. If the total
            number of input events requested for a given output file exceeds
            this number, the output will be split across multiple files.
        save_mc_events : bool
            Whether to save event data for all monte carlo showers, even for
            events which did not trigger the array (no images were saved).
        cleaning_settings : dict
            Settings for the cleaning used to calculate the parameters like
            Hillas, leakage etc.
        gain_selector_settings: dict
            Settings for the GainSelector.
        image_extractor_settings : dict
            Settings for the ImageExtractor for the calibration class.
        """
        self.event_source_class = event_source_class
        self.event_source_settings = (event_source_settings
                                      if event_source_settings else {})

        self.selected_telescope_ids = selected_telescope_ids

        self.data_dumper_class = data_dumper_class
        self.data_dumper_settings = (data_dumper_settings
                                     if data_dumper_settings else {})
        self.data_dumper_settings['save_mc_events'] = save_mc_events

        self.preselection_cut_function = preselection_cut_function

        if write_mode in ['serial', 'parallel']:
            self.write_mode = write_mode

        self.output_file_size = output_file_size
        self.events_per_file = events_per_file

        self.save_mc_events = save_mc_events

        if gain_selector_settings is None:
            gain_selector_settings = {'algorithm': 'ThresholdGainSelector', 'args': {'threshold': 3500}}
        self.gain_selector_settings = gain_selector_settings

        if image_extractor_settings is None:
            image_extractor_settings = {'algorithm': 'LocalPeakWindowSum', 'args': {'window_shift': 4, 'window_width': 8}}
        self.image_extractor_settings = image_extractor_settings

        if cleaning_settings is None:
            cleaning_settings = {'algorithm': 'tailcuts_clean', 'args': {'picture_thresh': 7, 'boundary_thresh': 5}}
        self.cleaning_settings = cleaning_settings

        if self.output_file_size:
            logger.info("Max output file size set at {} bytes. Note that "
                        "this may increase the number of output "
                        "files.".format(self.output_file_size))
        if self.events_per_file:
            logger.info("Max number of output events per file set at {}. Note "
                        "that this may increase the number of output "
                        "files.".format(self.events_per_file))


    def process_data(self, run_list):
        """Process data from a list of runs.
        If the selected write mode is parallel, creates one process for
        each requested run and executes them all in parallel.
        If the selected write mode is sequential, executes each run sequentially,
        writing each target one by one.
        Parameters
        ----------
        run_list : list of dicts
            A list of dictionaries, each containing two keys, 'inputs' and
            'target'. 'inputs' points to a list of input filenames (str) which
             are to be loaded. 'target' points to an output filename (str)
             to which the data from the input files should be written.
        """
        if self.write_mode == 'parallel':
            num_processes = len(run_list)
            logger.info("{} parallel processes requested.".format(num_processes))

            logger.info("Creating processes...")
            jobs = []
            for i in range(0, num_processes):
                process = multiprocessing.Process(target=self._process_data,
                                                  args=(run_list[i]['inputs'],
                                                        run_list[i]['target']))
                jobs.append(process)

            logger.info("Starting processes...")
            try:
                # Start all parallel processes
                for j in jobs:
                    j.start()

                # Wait for all processes to complete
                for j in jobs:
                    j.join()
            except KeyboardInterrupt:
                logger.error("Caught keyboard interrupt, killing all processes...")
                for j in jobs:
                    j.terminate()
        elif self.write_mode == 'serial':
            logger.info("Serial processing requested.")

            for run in run_list:
                logger.info("Starting run for target: {}...".format(run['target']))
                self._process_data(run['inputs'], run['target'])

        logger.info("Done!")

    @staticmethod
    def _get_next_filename(output_filename, output_file_count):
        """Get the next filename in the sequence.
        Parameters
        ----------
        output_filename : str
            The filename of the previous output file generated.
        output_file_count : int
            Number to attach to the current output file.
        Returns
        -------
        str
            Next filename in the sequence
        """
        # Append a trailing digit to get next filename in sequence
        dirname = os.path.dirname(output_filename)
        output_filename, *extensions = os.path.basename(
            output_filename).split('.')
        if re.search(r'_[0-9]+$', output_filename):
            output_filename = re.sub(
                r'_[0-9]+$', "_" + str(output_file_count),
                output_filename)
        else:
            output_filename = (
                output_filename + "_" + str(output_file_count))

        for ext in extensions:
            output_filename = output_filename + '.' + ext

        output_filename = os.path.join(dirname, output_filename)

        return output_filename

    def _process_data(self, file_list, output_filename):
        """Write a single output file given a list of input files.
        Parameters
        ----------
        file_list : list
            A list of input filenames (str) to read data from.
        output_filename : str
            Filename of the output file to write data to.
        """
        output_file_count = 1

        data_dumper = self.data_dumper_class(
            output_filename,
            **self.data_dumper_settings)
        filetype = "root"
        try:
            uproot.open(glob.glob(file_list[0])[0])
        except ValueError:
            # uproot raises ValueError if the file is not a ROOT file
            filetype = "simtel"

        for filename in file_list:
            if self.event_source_class:
                event_source = self.event_source_class(
                    filename,
                    **self.event_source_settings)
            elif filetype == "root":
                event_source = dl_eventsources.DLMAGICEventSource(input_url=filename)
            elif filetype == "simtel":
                event_source = io.eventsource.EventSource.from_url(filename, back_seekable = True)

            # Write all file-level data if not present
            # Or compare to existing data if already in file
            example_event = next(event_source._generator())

            subarray = event_source.subarray
            if filetype == "simtel":
                image_extractor = getattr(ctapipe.image.extractor, self.image_extractor_settings['algorithm'])(subarray=subarray,config=Config(self.image_extractor_settings['args']))
                calibrator = calib.camera.calibrator.CameraCalibrator(subarray=subarray, image_extractor=image_extractor)
            if self.selected_telescope_ids:
                subarray = subarray.select_subarray("subarray_for_selected_telids",self.selected_telescope_ids)
            mcheader = example_event.mcheader
            data_dumper.prepare_file(filename, subarray, mcheader, self.cleaning_settings)

            gain_selector = getattr(ctapipe.calib.camera.gainselection, self.gain_selector_settings['algorithm'])(config=Config(self.gain_selector_settings['args']))

            # Write all events sequentially
            for event in event_source:
                tels_id = event.r1.tels_with_data

                if filetype == "simtel":
                    for tel_id in tels_id:
                        if tel_id not in self.selected_telescope_ids:
                            event.r1.tel[tel_id].selected_gain_channel = gain_selector(event.r0.tel[tel_id].waveform)
                    calibrator(event)

                for tel_id in tels_id:
                    if tel_id not in self.selected_telescope_ids or not (math.isnan(event.dl1.tel[tel_id].parameters.hillas.intensity)):
                        continue

                    cleaning_method = getattr(cleaning, self.cleaning_settings['algorithm'])
                    cleanmask = cleaning_method(subarray.tel[tel_id].camera.geometry,
                                                    event.dl1.tel[tel_id].image,
                                                    **self.cleaning_settings['args'])
                    event.dl1.tel[tel_id].image_mask = cleanmask

                    if any(cleanmask):
                        leakage_values = leakage(subarray.tel[tel_id].camera.geometry,
                                                 event.dl1.tel[tel_id].image,
                                                 cleanmask)
                        hillas_parameters_values = hillas_parameters(subarray.tel[tel_id].camera.geometry[cleanmask],
                                                                     event.dl1.tel[tel_id].image[cleanmask])
                        concentration_values = concentration(subarray.tel[tel_id].camera.geometry,
                                                             event.dl1.tel[tel_id].image,
                                                             hillas_parameters_values)
                        try:
                            timing_values = timing_parameters(subarray.tel[tel_id].camera.geometry,
                                                              event.dl1.tel[tel_id].image,
                                                              event.dl1.tel[tel_id].peak_time,
                                                              hillas_parameters_values,
                                                              cleanmask)
                        except:
                            timing_values = containers.TimingParametersContainer()

                        morphology_values = morphology_parameters(subarray.tel[tel_id].camera.geometry, cleanmask)

                    else:
                        leakage_values = containers.LeakageContainer()
                        hillas_parameters_values = containers.HillasParametersContainer()
                        concentration_values = containers.ConcentrationContainer()
                        timing_values = containers.TimingParametersContainer()
                        morphology_values = containers.MorphologyContainer()

                    # Write parameter containers to the DataContainer
                    event.dl1.tel[tel_id].parameters.leakage = leakage_values
                    event.dl1.tel[tel_id].parameters.hillas = hillas_parameters_values
                    event.dl1.tel[tel_id].parameters.concentration = concentration_values
                    event.dl1.tel[tel_id].parameters.timing = timing_values
                    event.dl1.tel[tel_id].parameters.morphology = morphology_values

                if (self.preselection_cut_function is not None and not
                        self.preselection_cut_function(event)):
                    continue
                try:
                    data_dumper.dump_event(event)
                except IOError:
                    logger.error("Failed to write event from file "
                                 "{}, skipping...".format(
                                     filename))
                    break

                max_events_reached = ((self.events_per_file is not None) and (
                        data_dumper.event_index - 1 >= self.events_per_file))

                max_size_reached = ((self.output_file_size is not None) and (
                        os.path.getsize(
                            data_dumper.output_filename) > self.output_file_size))

                if max_events_reached or max_size_reached:
                    # Reset event count and increment file count
                    output_file_count += 1

                    output_filename = self._get_next_filename(
                        output_filename,
                        output_file_count)

                    # Create a new Data Dumper pointing at a new file
                    # and write file-level data
                    # Will flush + finalize + close file owned by
                    # previous data dumper
                    data_dumper = self.data_dumper_class(
                        output_filename, **self.data_dumper_settings)

                    # Write all file-level data if not present
                    # Or compare to existing data if already in file
                    data_dumper.prepare_file(filename, subarray, mcheader, self.cleaning_settings)

            if self.save_mc_events:
                for mc_event in event_source.file_.iter_mc_events():

                    try:
                        data_dumper.dump_mc_event(mc_event, event_source.file_.header['run'])
                    except IOError:
                        logger.error("Failed to write event from file "
                                     "{}, skipping...".format(
                            filename))
                        break

                    # Check whether to create another file
                    max_events_reached = ((self.events_per_file is not None) and (
                            data_dumper.event_index - 1 >= self.events_per_file))

                    max_size_reached = ((self.output_file_size is not None) and (
                            os.path.getsize(
                                data_dumper.output_filename) > self.output_file_size))

                    if max_events_reached or max_size_reached:
                        # Reset event count and increment file count
                        output_file_count += 1

                        output_filename = self._get_next_filename(
                            output_filename,
                            output_file_count)

                        # Create a new Data Dumper pointing at a new file
                        # and write file-level data
                        data_dumper = self.data_dumper_class(
                            output_filename, **self.data_dumper_settings)

                        # Write all file-level data if not present
                        # Or compare to existing data if already in file
                        data_dumper.prepare_file(filename, subarray, mcheader, self.cleaning_settings)
