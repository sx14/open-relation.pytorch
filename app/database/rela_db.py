import sqlite3


class RelaDB:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        # self.init_table()

    def init_table(self):
        try:
            create_tb_sql = '''
               CREATE TABLE IF NOT EXISTS relations
               (rela_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL ,
               sbj_id INTEGER NOT NULL,
               pre_id INTEGER NOT NULL,
               obj_id INTEGER NOT NULL,
               image_id CHAR(50) NOT NULL);
               '''

            create_index_sql = '''
                CREATE INDEX rela_index
                on relations (sbj_id, pre_id, obj_id);
                '''
            self.cursor.execute(create_tb_sql)
            self.cursor.execute(create_index_sql)
        except:
            print("Create table failed")
            return False

    def insert_rela(self, rela_image_tuples):
        sql = 'insert into relations (sbj_id, pre_id, obj_id, image_id) values (?,?,?,?)'
        self.cursor.executemany(sql, rela_image_tuples)
        self.conn.commit()

    def find_images_by_rela(self, sbjs, pres, objs):
        sql = 'select image_id from relations where sbj_id in ({0}) AND pre_id in ({1}) AND obj_id in ({2})'.format(
            ', '.join('?' for _ in sbjs), ', '.join('?' for _ in pres), ', '.join('?' for _ in objs))
        print(sql)
        try:
            self.cursor.execute(sql, sbjs + pres + objs)
            rows = self.cursor.fetchall()
            return rows
        except:
            self.close()
            return []

    def stat_rela(self, concepts):
        sql = 'select pre_id, obj_id, count(*) as rela_num from relations where sbj_id in ({0}) group by pre_id, obj_id ORDER BY rela_num DESC'.format(
            ', '.join('?' for _ in concepts))
        try:
            self.cursor.execute(sql,concepts)
            rows = self.cursor.fetchall()
            return rows
        except:
            self.close()
            return []

    def close(self):
        self.cursor.close()
        self.conn.close()
