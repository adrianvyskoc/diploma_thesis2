import pandas as pd
import psycopg2

try:
    connection = psycopg2.connect(
        user = "adrianvyskoc",
        password = "12345",
        host = "127.0.0.1",
        port = "5432",
        database = "diploma_thesis"
    )

    cursor_1 = connection.cursor()
    cursor_2 = connection.cursor()
    cursor_3 = connection.cursor()
    cursor_4 = connection.cursor()
    cursor_5 = connection.cursor()
    
    print(connection.get_dsn_parameters(),"\n")

    # Print PostgreSQL version
    cursor_1.execute("SELECT version();")
    record = cursor_1.fetchone()
    print("You are connected to - ", record,"\n")

except (Exception, psycopg2.Error) as error :
    print ("Error while connecting to PostgreSQL", error)
    


# get data from a database
cursor_1.execute("""
    SELECT 
        adm."AIS_ID",
        adm."school_id",
        adm."Body", 
        adm."Exb_celk", 
        adm."Maturita", 
        adm."Najvyššie_dosiahnuté_vzdelanie", 
        adm."Pohlavie", 
        adm."Občianstvo", 
        adm."Odkiaľ_sa_hlási", 
        adm."Program", 
        adm."Narodenie",
        adm."Maturita_1" as graduation_year,
        adm."stupen_studia",
        adm."Odvolanie",
        std2."Typ_ukončenia",
		std2."VŠP_štúdium"
    FROM ais_admissions as adm
    JOIN ais_students_data_pt_2 as std2
    ON std2."ID" = adm."AIS_ID"
    WHERE 
		-- tato subquery zaručí, že ak mame importovane data o studentoch viac krat, zoberie tie zaznamy, ktore su z najnizsieho skolskeho roka
		(std2."ID", std2."OBDOBIE") IN (
			SELECT std."ID", MIN(std."OBDOBIE") FROM ais_students_data_pt_2 as std
			GROUP BY std."ID"
		) AND 
		--adm."OBDOBIE" = '2014-2015' AND
		adm."AIS_ID" IS NOT NULL AND 
		adm.Štúdium = 'áno' AND 
		std2."Typ_ukončenia" IS NOT NULL;
""")

cursor_2.execute("""
    SELECT * FROM ais_grades as grd 
    JOIN ais_subjects as sbj ON grd."PREDMET_ID" = sbj.id
""")

cursor_3.execute("""
    SELECT
		grd."AIS_ID", 
		COUNT(grd."PREDMET_VYSLEDOK") as studied_subjects_count,
		SUM(
			CASE
			WHEN grd."PREDMET_VYSLEDOK" = NULL THEN 0
			WHEN grd."PREDMET_VYSLEDOK" = 'FN' THEN 0
			WHEN grd."PREDMET_VYSLEDOK" = 'FX' THEN 0
			ELSE 1
			END
		) as passed,
		SUM(
			CASE
			WHEN grd."PREDMET_VYSLEDOK" = NULL THEN 1
			WHEN grd."PREDMET_VYSLEDOK" = 'FN' THEN 1
			WHEN grd."PREDMET_VYSLEDOK" = 'FX' THEN 1
			ELSE 0
			END
		) as failed
	FROM ais_grades as grd
	GROUP BY grd."AIS_ID";
""")

cursor_4.execute("""
    SELECT atd."AIS_ID", sbj."KOD", count(atd."UCAST_ID") FROM ais_attendances as atd 
    JOIN ais_subjects as sbj ON atd."PREDMET_ID" = sbj.id
    WHERE atd."UCAST_ID" = 1
    GROUP BY atd."AIS_ID", sbj."KOD"
""")

cursor_5.execute("""
    SELECT
        ineko_total_ratings."school_id",
        ineko_total_ratings."celkove_hodnotenie" AS school_general_rating,
        ineko_total_ratings."matematika" AS school_math_rating,
        ineko_total_ratings."cudzie_jazyky" AS school_foreign_lang_rating,
        ineko_total_ratings."prijimanie_na_VS" AS school_university_admissions_rating,
        ineko_total_ratings."financne_zdroje" AS school_finance_sources_rating,
        ineko_total_ratings."pedagogicky_zbor" AS school_pedagogical_team
    FROM ineko_total_ratings
""")

# save data to dataframe
admissions = pd.DataFrame(data = cursor_1.fetchall(), columns = [data.name for data in cursor_1.description])
grades = pd.DataFrame(data = cursor_2.fetchall(), columns = [data.name for data in cursor_2.description])
grades_counts = pd.DataFrame(data = cursor_3.fetchall(), columns = [data.name for data in cursor_3.description])
attendances = pd.DataFrame(data = cursor_4.fetchall(), columns = [data.name for data in cursor_4.description])
school_total_ratings = pd.DataFrame(data = cursor_5.fetchall(), columns = [data.name for data in cursor_5.description])

# close database connection 
cursor_1.close()
cursor_2.close()
cursor_3.close()
connection.close()