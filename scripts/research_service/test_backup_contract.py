import hashlib
import json
import pathlib
import sqlite3
import subprocess
import tempfile
import unittest
import backup_contract as bc


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        base=pathlib.Path(self.tmp.name);self.root=base/'repo';self.root.mkdir();self.state=base/'state';self.state.mkdir()
        self.remote=base/'external.git'
        subprocess.run(['git','init','--bare',str(self.remote)],check=True,capture_output=True)
        self.git('init');self.git('config','user.name','Test');self.git('config','user.email','test@example.invalid')
        (self.root/'controller.py').write_text('x=1\n');(self.root/'RESEARCH_LOG.md').write_text('Historical log\n')
        self.git('add','.');self.git('commit','-m','Fixture');self.git('remote','add','origin',str(self.remote));self.git('push','-u','origin','HEAD')
        commit=self.git('rev-parse','HEAD').stdout.decode().strip()
        self.status={'status':'verified','remaining_changed_files':0,'commit':commit,'packages_this_run':[{'sha256':'a'*64,'verification':'github_server_sha256'}]}
        (self.state/'backup_status.json').write_text(json.dumps(self.status))
        with sqlite3.connect(self.state/'backup_index.sqlite') as c:c.execute('create table files(path text primary key,sha text,asset text)')
    def git(self,*args):return subprocess.run(['git',*args],cwd=self.root,capture_output=True,check=True)
    def test_live_log_changes_do_not_invalidate_backed_immutable_code(self):
        (self.root/'RESEARCH_LOG.md').write_text('New live append after backup\n')
        result=bc.verify(self.root,self.state,['controller.py'])
        self.assertTrue(result['verified_before_solver_or_plant_steps'])
    def test_changed_unpushed_scientific_source_is_rejected(self):
        (self.root/'controller.py').write_text('x=2\n')
        with self.assertRaises(ValueError):bc.verify(self.root,self.state,['controller.py'])
    def test_verified_raw_release_index_is_checked_by_content(self):
        p=self.root/'raw.json';p.write_text('{"value":1}')
        with sqlite3.connect(self.state/'backup_index.sqlite') as c:c.execute('insert into files values(?,?,?)',('repo/raw.json',bc.sha(p),'verified-package.tar.gz'))
        self.assertTrue(bc.verify(self.root,self.state,['raw.json'])['verified_before_solver_or_plant_steps'])
        p.write_text('{"value":2}')
        with self.assertRaises(ValueError):bc.verify(self.root,self.state,['raw.json'])
    def test_failed_backup_or_missing_digest_is_rejected(self):
        for field,value in [('status','failed'),('packages_this_run',[])]:
            status=dict(self.status);status[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):bc.validate_status(status)
    def test_outside_repository_or_missing_file_is_rejected(self):
        for path in ['../state/backup_status.json','missing.py']:
            with self.subTest(path=path),self.assertRaises(ValueError):bc.verify(self.root,self.state,[path])


if __name__=='__main__':unittest.main()
