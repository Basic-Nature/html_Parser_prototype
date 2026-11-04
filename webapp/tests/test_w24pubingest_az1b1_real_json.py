"""W24PUBINGEST-AZ1B1 tests over the user's unmodified real Arizona snapshots.

Run: python tests/test_w24pubingest_az1b1_real_json.py --input-zip PATH
No network, no repository change, no vote data fabrication.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import zipfile

HERE = Path(__file__).resolve().parent.parent
ADAPTER = HERE / 'parser/utils/enhanced_voting_precinct_adapter.py'
SPEC = importlib.util.spec_from_file_location('ev_az1b1_adapter', ADAPTER)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

ZIP_SHA256 = 'c4000266d0452b97e0aec70458d261de7b96d5e5b2d0a40e3c38ebeffb549422'
LATEST_SHA256 = '35ab7ce79145c767e80353702f0e0439037cd50152946fd90b6b0ac36583c63b'
SOURCE_METHODS = {'Election Day','Early Voting','Provisional','Early A.R.S. 16-579','Late Early*'}
ZIP_PATH = None


def source_snapshots():
    raw = ZIP_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ZIP_SHA256, 'Unexpected original ZIP bytes'
    with zipfile.ZipFile(ZIP_PATH) as z:
        assert z.testzip() is None, 'ZIP CRC failure'
        entries = []
        for member in z.infolist():
            if member.filename.lower().endswith('.json'):
                assert member.file_size <= 130*1024*1024
                b = z.read(member)
                d = json.loads(b)
                entries.append((d['createdAt'], member.filename, hashlib.sha256(b).hexdigest(), d))
        assert len(entries) == 5
        return sorted(entries)


class RealSourceQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshots = source_snapshots()
        cls.latest = cls.snapshots[-1][3]

    def test_source_file_and_identity(self):
        self.assertEqual(self.snapshots[-1][2], LATEST_SHA256)
        self.assertEqual(self.latest['results']['name'], 'Arizona')
        self.assertEqual(self.latest['electionDate'], '2026-11-03')
        self.assertEqual(len(self.latest['localResults']),15)
        self.assertEqual(len(self.latest['results']['ballotItems']),99)

    def test_all_county_contests(self):
        total_precincts = 0
        expected = 0
        for county in self.latest['localResults']:
            for contest in county['ballotItems']:
                expected += 1
                got = module.build_precinct_table(self.latest,contest_id=str(contest['id']),county=str(county['id']))
                self.assertEqual(got.metadata['issue_count'],0,(county['name'],contest['id'],got.metadata['issues'][:5]))
                self.assertTrue(got.metadata['summary_row_explicitly_non_precinct'])
                self.assertEqual(got.rows[-1]['Discrepancy Flag'],'SUMMARY_NOT_PRECINCT')
                precinct_ids = {p['id'] for o in contest['ballotOptions'] for p in o.get('precinctResults',[])}
                self.assertEqual(len(got.rows)-1,len(precinct_ids))
                self.assertEqual({r['Precinct ID'] for r in got.rows[:-1]},set(map(str,precinct_ids)))
                expected_total = sum(o['voteCount'] for o in contest['ballotOptions'])
                self.assertEqual(got.rows[-1]['Grand Total'],expected_total)
                total_precincts+=len(got.rows)-1
        self.assertEqual(expected,415)
        self.assertEqual(total_precincts,42770)

    def test_three_counties_az10(self):
        expected = {'Yavapai County':(57,88650),'Maricopa County':(956,1004925),'Graham County':(23,0)}
        for county,(precinct_count,votes) in expected.items():
            table=module.build_precinct_table(self.latest,contest_id='AZ-10',county=county)
            self.assertEqual(len(table.rows)-1,precinct_count)
            self.assertEqual(table.rows[-1]['Grand Total'],votes)
            self.assertEqual(table.metadata['issue_count'],0)
            self.assertEqual(set(table.metadata['source_methods']),SOURCE_METHODS)
            self.assertIn('Absentee Mail',table.metadata['unreported_standard_methods'])
            self.assertTrue(all(row['Absentee Mail Total'] is None for row in table.rows[:-1]))
            self.assertEqual(len(table.metadata['candidate_ids']),8)

    def test_all_county_governor_every_snapshot(self):
        expected = [0,0,1584769,1597072,1598133]
        for (created,name,sha,d),total in zip(self.snapshots,expected):
            table=module.build_precinct_table(d,contest_id='AZ-10',source_sha256=sha)
            self.assertEqual(len(table.rows)-1,1762)
            self.assertEqual(table.rows[-1]['Grand Total'],total)
            self.assertEqual(table.metadata['issue_count'],0)
            self.assertEqual(table.metadata['snapshot_created_at'],created)
            self.assertTrue(all(row['Snapshot Created At']==created for row in table.rows[:-1]))
            self.assertTrue(all(row['Source SHA256']==sha for row in table.rows[:-1]))

    def test_explicit_zero_vs_absent_and_reporting(self):
        first=module.build_precinct_table(self.snapshots[0][3],contest_id='AZ-10',county='Graham County')
        self.assertTrue(any(row['Grand Total']==0 for row in first.rows[:-1]))
        for row in first.rows[:-1]:
            self.assertIsNone(row['Absentee Mail Total'])
            self.assertIsInstance(row['Reporting Status'],str)
            self.assertNotEqual(row['Reporting Status'],'')
        latest=module.build_precinct_table(self.latest,contest_id='AZ-10',county='Maricopa County')
        self.assertTrue(any(row['Grand Total']>0 and row['Reporting Status']=='Not Reported' for row in latest.rows[:-1]))

    def test_no_ambiguous_selection(self):
        with self.assertRaises(module.EnhancedVotingPrecinctError):
            module.build_precinct_table(self.latest,contest_id='NO-SUCH-CONTEST')
        with self.assertRaises(module.EnhancedVotingPrecinctError):
            module.build_precinct_table(self.latest,contest_id='AZ-10',county='Not a County')

    def test_missing_method_remains_null(self):
        import copy
        d=copy.deepcopy(self.latest)
        county=next(c for c in d['localResults'] if c['name']=='Graham County')
        opt=next(o for t in county['ballotItems'] if t['id']=='AZ-10' for o in t['ballotOptions'])
        p=opt['precinctResults'][0]
        original=p['groupResults'][0]
        method=original['groupName']
        p['groupResults'].pop(0)
        table=module.build_precinct_table(d,contest_id='AZ-10',county='Graham County')
        row=next(r for r in table.rows[:-1] if r['Precinct ID']==str(p['id']))
        col=f"{opt['name']} - {method}"
        self.assertIsNone(row[col])
        self.assertIsNone(row[f'{method} Total'])
        self.assertEqual(row['Grand Total'],0)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input-zip',required=True);args=parser.parse_args()
    ZIP_PATH=Path(args.input_zip)
    unittest.main(argv=[sys.argv[0]],verbosity=2)
