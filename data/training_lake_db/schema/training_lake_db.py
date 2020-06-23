from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, NVARCHAR, BigInteger, Float, VARCHAR

Model = declarative_base()
COLLATION = "SQL_Latin1_General_CP1_CI_AS"


class Detail(Model):
	__tablename__ = 'Detail'
	run_id = Column(NVARCHAR(length=100, collation=COLLATION), primary_key=True)
	name = Column(NVARCHAR(length=None, collation=COLLATION))
	value = Column(NVARCHAR(length=None, collation=COLLATION))
	type = Column(NVARCHAR(length=None, collation=COLLATION))

	def __repr__(self):
		return f"<{self.__tablename__}(run_id={self.run_id}, name={self.name}, value={self.value}, type={self.type})>"
