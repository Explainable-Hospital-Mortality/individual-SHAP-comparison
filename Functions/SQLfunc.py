# IMPORT
import pandas as pd
from Functions.util import getCategory, nonOperative, operative

# FUNCTIONS
def getPatientUnitStayIDString(patientunitstayids):
    if isinstance(patientunitstayids, int):
        return 'patientunitstayid IN (' + str(patientunitstayids) + ')'
    s = 'patientunitstayid IN (' + str(patientunitstayids[0])
    if len(patientunitstayids) > 1:
        for id in patientunitstayids[1:]:
            s = s + ', ' + str(id)
    s = s + ')'
    return s


def getPatientsWithMoreThanOneICUStay(conn, query_schema):
    query = query_schema + \
        """
        SELECT DISTINCT(uniquepid), COUNT(uniquepid)
        FROM patient
        GROUP BY uniquepid
        HAVING COUNT(uniquepid) > 1
        ORDER BY uniquepid
        """
    return pd.read_sql_query(query, conn)['uniquepid'].values


def getAPACHE(query_schema, conn, patientunitstayids, apacheVersion):
    query = query_schema + \
        """
        SELECT patientUnitStayID, acutePhysiologyScore, apacheVersion, predictedICUMortality, actualICUMortality, predictedHospitalMortality, actualHospitalMortality
        FROM apachePatientResult
        WHERE """ + getPatientUnitStayIDString(patientunitstayids) + """ AND predictedhospitalmortality != '-1'
        ORDER BY patientunitstayid
        """
    apacheResults = pd.read_sql_query(query, conn)

    apacheResults.loc[apacheResults['actualicumortality'] == 'ALIVE', 'actualicumortality'] = 0
    apacheResults.loc[apacheResults['actualicumortality'] == 'EXPIRED', 'actualicumortality'] = 1
    apacheResults.actualicumortality = apacheResults.actualicumortality.astype(int)
    apacheResults.predictedicumortality = apacheResults.predictedicumortality.astype(float)

    apacheResults.loc[apacheResults['actualhospitalmortality'] == 'ALIVE', 'actualhospitalmortality'] = 0
    apacheResults.loc[apacheResults['actualhospitalmortality'] == 'EXPIRED', 'actualhospitalmortality'] = 1
    apacheResults.actualhospitalmortality = apacheResults.actualhospitalmortality.astype(int)
    apacheResults.predictedhospitalmortality = apacheResults.predictedhospitalmortality.astype(float)

    return apacheResults[apacheResults.apacheversion == apacheVersion]


def getAPACHEPredVar(query_schema, conn, patientunitstayids):
    query = query_schema + \
        """
        SELECT *
        FROM apachePredVar
        WHERE """ + getPatientUnitStayIDString(patientunitstayids) + """
        ORDER BY patientunitstayid
        """
    return pd.read_sql_query(query, conn)


def getAPS(query_schema, conn, patientunitstayids):
    query = query_schema + \
        """
        SELECT *
        FROM apacheApsVar
        WHERE """ + getPatientUnitStayIDString(patientunitstayids) + """
        ORDER BY patientunitstayid
        """
    return pd.read_sql_query(query, conn)


def getAdx(conn, query_schema):
    query = query_schema + \
                    """
                    SELECT patientunitstayid
                    , admitDxPath
                    FROM admissiondx
                    ORDER BY patientunitstayid
                    """

    adx = pd.read_sql_query(query, conn)

    adx['split'] = adx['admitdxpath'].apply(lambda x: x.split('|'))
    adx['category'] = adx['split'].apply(lambda x: getCategory(x))
    adx['nonOperative'] = adx['category'].apply(lambda x: nonOperative(x))
    adx['operative'] = adx['category'].apply(lambda x: operative(x))
    nonOp = adx[~adx['nonOperative'].isnull()]
    op = adx[~adx['operative'].isnull()]

    return pd.merge(nonOp[['patientunitstayid', 'nonOperative']], op[['patientunitstayid', 'operative']], on='patientunitstayid', how='outer')


def getExpiredPatients(conn, query_schema, moreThanOneStay, startAge, endAge, minLoS):
    query = query_schema + \
        """
        SELECT patientunitstayid
        , uniquepid
        , age
        , gender
        , ethnicity
        , admissionHeight
        , admissionWeight
        , unitDischargeOffset
        , hospitalAdmitOffset
        , hospitaldischargestatus
        , unitdischargestatus
        , unitAdmitSource
        , unitType
        , unitStayType
        , hospitalID
        FROM patient
        WHERE ((hospitaldischargestatus = 'Expired') OR (unitVisitNumber = 1 AND unitDischargeStatus = 'Expired'))
        ORDER BY patientunitstayid
        """

    return  pd.read_sql_query(query, conn)


def getAlivePatients(conn, query_schema, moreThanOneStay, startAge, endAge, minLoS):
    query = query_schema + \
        """
        SELECT patientunitstayid
        , uniquepid
        , age
        , gender
        , ethnicity
        , admissionHeight
        , admissionWeight
        , unitDischargeOffset
        , hospitalAdmitOffset
        , hospitaldischargestatus
        , unitdischargestatus
        , unitAdmitSource
        , unitType
        , unitStayType
        , hospitalID
        FROM patient
        WHERE unitVisitNumber = 1 AND unitDischargeStatus = 'Alive' AND hospitalDischargeStatus = 'Alive'
        ORDER BY patientunitstayid
        """

    return pd.read_sql_query(query, conn)
