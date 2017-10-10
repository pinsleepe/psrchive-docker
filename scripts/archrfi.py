import matplotlib as mpl
# for Docker use
mpl.use('Agg')

from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt

import pysolr
import collections
import sys
import requests
from os.path import join, isfile, exists, getsize
from os import listdir, makedirs
import requests
import psrchive as psr
from coast_guard import cleaners
import time
import h5py
import numpy as np
from shutil import rmtree


class ObsQuery(object):
    def __init__(self, product='PTUSETimingArchiveProduct', verbose=False):
        self.product = product
        self.search = None
        self.solr_url = 'http://192.168.1.50:8983/solr/kat_core'
        self.results = None
        self.verbose = verbose

    def standard_observation_query(self, days):
        """

        :param days:
        :param product: PulsarTimingArchiveProduct, PTUSETimingArchiveProduct, PulsarSearchProduct
        :return:
        """
        # Create a quick field+query type
        fq = collections.namedtuple('field_query', 'field query')

        query_list = [
            # Only want MeerKAT AR1 Telescope Products
            fq('CAS.ProductTypeName', self.product),
            # Observations from the last 3 days
            fq('StartTime', '[NOW-%dDAYS TO NOW]' % days)]

        # Construct the query
        return ' AND '.join('%s:%s' % (fq.field, fq.query)
                            for fq in query_list)

    def query_recent_observations(self, days=3, query=None):
        self.search = query if query is not None else self.standard_observation_query(days)
        if self.verbose:
            print("Querying solr server '%s' with query string '%s'." % (self.solr_url, self.search))

        archive = pysolr.Solr(self.solr_url)

        self.results = archive.search(self.search, sort='StartTime desc', rows=1000)
        if self.verbose:
            print("Found %d matching results." % self.results.hits)


class PulsarArchive(object):
    def __init__(self, observation, verbose=False):
        self.observation = observation
        self.verbose = verbose
        self.archive_name = observation['Filename']
        self.remote_location = None
        self.arch_files_remote = None
        self.arch_files_local = []
        self.arch_urls = None
        self.ar_files = None
        self.psr_archive = None
        self.weights = None
        self.mask = None
        self.lowFreq = None
        self.highFreq = None
        self.sourceName = None
        self.nChan = None
        self.rfi_occupancy = None
        self.local_archive_path = None

        self._init()

    def _init(self):
        self._get_archives_list()
        self._get_ar_list()
        self._get_archives_urls()

    def _get_archives_list(self):
        archive_remote_files = self.observation['CAS.ReferenceDatastore'][1:-1]
        self.arch_files_remote = [f for f in archive_remote_files if f.split('.')[1] == 'ar']

    def _get_ar_list(self):
        self.ar_files = [l.split('/')[-1] for l in self.arch_files_remote]
        self.ar_files.sort()

    def _get_archives_urls(self):
        remote_location = self.observation['FileLocation'][0]
        remote_location = remote_location.replace('/var/kat', 'http://kat-archive.kat.ac.za', 1)
        self.remote_location = remote_location.replace('archive/', 'archive2/', 1)
        self.arch_urls = [join(remote_location, self.archive_name, a) for a in self.ar_files]
        self.arch_urls.sort()

    def _writeDir(self, full_path):
        if not exists(full_path):
            makedirs(full_path)
        # check if successful
        return exists(full_path)

    def download_observation(self, directory):

        ONE_KB = 1024
        ONE_MB = ONE_KB * ONE_KB
        if self.verbose:
            print(self.archive_name)

        # create obs folder
        local_archive_path = join(directory, self.archive_name)
        self.local_archive_path = local_archive_path
        if self.verbose:
            print(local_archive_path)

        # check if directory exists, if not create one
        self._writeDir(local_archive_path)
        # loop through arch files
        for idx, url in enumerate(self.arch_urls):
            filename = join(local_archive_path, self.ar_files[idx])
            self.arch_files_local.append(filename)
            file_exists = exists(filename) and isfile(filename)
            local_file_size = getsize(filename) if file_exists else 0
            # Infer the HTTP location from the KAT archive file location
            headers = {"Range": "bytes={}-".format(local_file_size)}
            if file_exists:
                r = requests.get(url, headers=headers, stream=True)
            else:
                r = requests.get(url, stream=True)
            # Server doesn't care about range requests and is just
            # sending the entire file
            if r.status_code == 200:
                if self.verbose:
                    print("Downloading '{}'")
                remote_file_size = r.headers.get('Content-Length', None)
                file_exists = False
                local_file_size = 0
            elif r.status_code == 206:
                if local_file_size > 0:
                    if self.verbose:
                        print("'{}' already exists, resuming download from {}.".format(filename,
                                                                                   local_file_size))

                # Create a fake range if none exists
                fake_range = "{}-{}/{}".format(local_file_size, sys.maxint,
                                               sys.maxint - local_file_size)

                remote_file_size = r.headers.get('Content-Range', fake_range)
                remote_file_size = remote_file_size.split('/')[-1]
            elif r.status_code == 416:
                if self.verbose:
                    print("'{}' already downloaded".format(filename))
                remote_file_size = local_file_size
            else:
                raise ValueError("HTTP Error Code {}".format(r.status_code))
            if self.verbose:
                print('%s %s %s' % (url, remote_file_size, r.status_code))

            f = open(filename, 'ab' if file_exists else 'wb')

            # Download chunks of file and write to disk
            try:
                with f:
                    downloaded = local_file_size
                    for chunk in r.iter_content(chunk_size=ONE_MB):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if self.verbose:
                                print(downloaded)
            except KeyboardInterrupt as kbe:
                pass
                if self.verbose:
                    print("Quitting download on Keyboard Interrupt")

    def gen_archive(self):
        self.arch_files_local.sort()
        archives = [psr.Archive_load(f.encode('ascii', 'ignore'))
                    for f in self.arch_files_local]
        for i in range(1, len(archives)):
            archives[0].append(archives[i])
        archives[0].pscrunch()
        self.psr_archive = archives[0]
        self.lowFreq = self.psr_archive.get_centre_frequency() - self.psr_archive.get_bandwidth() / 2.0
        self.highFreq = self.psr_archive.get_centre_frequency() + self.psr_archive.get_bandwidth() / 2.0
        self.sourceName = self.psr_archive.get_source()
        self.nChan = self.psr_archive.get_nchan()

    def gen_weights(self):
        """
        get weights
        """
        cleaner = cleaners.load_cleaner("surgical")
        cleaner.parse_config_string("chan_numpieces=1,"
                                    "subint_numpieces=1,"
                                    "chanthresh=3,"
                                    "subintthresh=3")
        cleaner.run(self.psr_archive)
        self.weights = self.psr_archive.get_weights().T

    def cut_edges(self):
        cleaner2 = cleaners.load_cleaner('rcvrstd')
        rcvrstd_parameters = 'badfreqs=None,' \
                             'badsubints=None,' \
                             'trimbw=0,' \
                             'trimfrac=0,' \
                             'trimnum=0,' \
                             'response=None,' \
                             'badchans=0:210;3896:4095'
        cleaner2.parse_config_string(rcvrstd_parameters)
        cleaner2.run(self.psr_archive)
        self.mask = self.psr_archive.get_weights().T
        self.count_rfi()

    def count_rfi(self):
        """
        percentage of rfi channels
        """
        count = np.where(self.mask == 0)[1].shape[0]
        self.rfi_occupancy = (float(count)/float(self.mask.size))*100

    def plot_mask(self, array, save=False, directory=''):
        fig, ax1 = plt.subplots(1, 1,
                                figsize=[15, 10],
                                tight_layout="false")
        ax1.imshow(array,
                   origin="lower",
                   aspect="auto")
        ax1.set_title(self.archive_name)
        ax1.set_title("RFI mask", loc="left")
        ax1.set_ylabel("Channel number")
        ax1.yaxis.set_ticks(np.arange(0, self.nChan - 1, 200))
        ax1.set_xlabel("Subint number")
        ax1Secondary = ax1.twinx()
        ax1Secondary.set_ylabel("Frequency (MHz)")
        ax1Secondary.set_ylim(self.lowFreq, self.highFreq)
        ax1Secondary.yaxis.set_ticks(np.arange(self.lowFreq, self.highFreq, 25))
        if save:
            fig.savefig(join(directory, self.archive_name + '.png'))
        return fig

    def write_h5(self, directory=''):
        with h5py.File(join(directory, self.archive_name + '.h5'), 'w') as hf:
            dset = hf.create_dataset('folded_obs', data=self.mask, dtype='int8')
            dset.attrs['nChan'] = self.nChan
            dset.attrs['nSubint'] = self.psr_archive.get_nsubint()
            dset.attrs['sourceName'] = self.archive_name
            dset.attrs['RA'] = self.psr_archive.get_coordinates().ra().getHMS()
            dset.attrs['Dec'] = self.psr_archive.get_coordinates().dec().getDMS()
            dset.attrs['centreFrequency'] = self.psr_archive.get_centre_frequency()
            dset.attrs['bandwidth'] = self.psr_archive.get_bandwidth()
            dset.attrs['DM'] = self.psr_archive.get_dispersion_measure()
            dset.attrs['obsDuration'] = self.psr_archive.integration_length()
            dset.attrs['rfi_percentage'] = self.rfi_occupancy

    def cleanup(self):
        """
        short fix, use multiprocessing
        """
        self.arch_files_remote = None
        self.arch_files_local = []
        self.arch_urls = None
        self.ar_files = None
        self.psr_archive = None
        self.weights = None
        self.mask = None

        rmtree(self.local_archive_path)


class Timer:
    def __init__(self):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def get_time_hhmmss(self):
        end = time.time()
        m, s = divmod(end - self.start, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        return time_str

    def print_time(self, time_elapsed):
        print("Time elapsed: %s" % time_elapsed)


if __name__ == "__main__":
    my_timer = Timer()
    obs_query = ObsQuery()
    days = 10
    obs_query.query_recent_observations(days)
    for obs in obs_query.results.docs:
        archive = PulsarArchive(obs)
        local_directory = '/data'
        archive.download_observation(local_directory)
        archive.gen_archive()
        archive.gen_weights()
        archive.cut_edges()
        fig = archive.plot_mask(archive.mask, save=True, directory='/data')
        fig.clear()
        archive.write_h5(directory='/data')
        archive.cleanup()
    time_hhmmss = my_timer.get_time_hhmmss()
    my_timer.print_time(time_hhmmss)
