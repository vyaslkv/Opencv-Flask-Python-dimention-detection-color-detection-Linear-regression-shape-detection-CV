import MySQLdb
import os


def to_database():
    crops = os.listdir('static/crops/')

    for crop in crops:
        path = os.path.abspath(crop)
        db = MySQLdb.connect("localhost", "root", "password", "test")
        with db:
            cursor = db.cursor()

            sql = "INSERT INTO cv (img1) VALUES (%s)"

            cursor.execute(sql, (path,))

        db.close()