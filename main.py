import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def init_examen():
    """
    Función principal para la limpieza, exploración de datos y análisis adicional del dataset 'ds_salaries.csv'.
    """

    # 1. Limpieza de datos
    # Cargar el dataset
    df = pd.read_csv('ds_salaries.csv')

    # Detección y eliminación de valores duplicados
    duplicates = df.duplicated().sum()  # Contar duplicados antes de eliminarlos
    df = df.drop_duplicates()  # Eliminar duplicados

    # Verificación y ajuste de tipos de datos
    # Aseguramos que los tipos de datos sean correctos
    df['work_year'] = df['work_year'].astype(int)
    df['salary'] = df['salary'].astype(float)
    df['salary_in_usd'] = df['salary_in_usd'].astype(float)
    df['remote_ratio'] = df['remote_ratio'].astype(int)

    # Consistencia en valores categóricos
    # Convertimos a minúsculas para asegurar consistencia en los valores categóricos
    df['experience_level'] = df['experience_level'].str.lower()
    df['employment_type'] = df['employment_type'].str.lower()
    df['company_size'] = df['company_size'].str.lower()

    # Manejo de valores faltantes
    # Imputamos valores faltantes en las columnas categóricas
    missing_values = df.isnull().sum()  # Contamos los valores faltantes
    df.fillna({'salary_currency': 'Unknown', 'company_location': 'Unknown'}, inplace=True)

    # Detección de datos anómalos (outliers)
    # Utilizamos Z-score para identificar outliers en 'salary_in_usd'
    z_scores = np.abs(stats.zscore(df['salary_in_usd']))
    df = df[z_scores < 3]  # Eliminamos filas con Z-score mayor a 3 (posibles outliers)

    # Imprimimos el estado después de la limpieza
    print("Limpieza de datos completada.")
    print(f"Registros duplicados eliminados: {duplicates}")
    print(f"Valores faltantes por columna antes del llenado:\n{missing_values}")
    print(f"Registros después de eliminar outliers: {len(df)}")

    # 2. Exploración de datos

    # Visualizaciones exploratorias univariadas

    # Distribución de salarios en USD
    plt.figure(figsize=(10, 6))
    sns.histplot(df['salary_in_usd'], kde=True, color='skyblue')
    plt.title('Distribución de salarios en USD')
    plt.xlabel('Salario en USD')
    plt.ylabel('Frecuencia')
    plt.savefig('salarios_distribucion.png')
    plt.close()

    # Interpretación: La distribución de salarios muestra una asimetría positiva,
    # lo que indica que hay más salarios bajos y medios que salarios muy altos.

    # Distribución de niveles de experiencia
    plt.figure(figsize=(10, 6))
    df['experience_level'].value_counts().plot(kind='bar', color='coral')
    plt.title('Distribución de niveles de experiencia')
    plt.xlabel('Nivel de experiencia')
    plt.ylabel('Cantidad')
    plt.savefig('niveles_experiencia.png')
    plt.close()

    # Interpretación: La mayoría de los profesionales en el dataset tienen un nivel
    # de experiencia senior (se), seguido por nivel medio (mi) y entrada (en).

    # Visualizaciones exploratorias multivariadas

    # Salarios por nivel de experiencia (boxplot)
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x='experience_level',
        y='salary_in_usd',
        data=df,
        hue='experience_level',
        palette="Set2",
        dodge=False,
        legend=False
    )
    plt.title('Salarios por nivel de experiencia')
    plt.xlabel('Nivel de experiencia')
    plt.ylabel('Salario en USD')
    plt.savefig('salarios_por_experiencia.png')
    plt.close()

    # Interpretación: Los salarios aumentan con el nivel de experiencia, y los niveles
    # ejecutivos (ex) muestran la mayor variabilidad y los salarios más altos.

    # Salarios vs. Ratio de trabajo remoto por tamaño de empresa (scatterplot)
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='remote_ratio', y='salary_in_usd', hue='company_size', data=df, palette='coolwarm')
    plt.title('Salarios vs. Ratio de trabajo remoto por tamaño de empresa')
    plt.xlabel('Ratio de trabajo remoto')
    plt.ylabel('Salario en USD')
    plt.savefig('salarios_remoto_tamanio.png')
    plt.close()

    # Interpretación: No parece haber una clara relación entre el ratio de trabajo remoto
    # y el salario. Sin embargo, las empresas grandes tienden a ofrecer salarios más altos.

    # 3. Análisis adicional

    # Estadísticas descriptivas para la columna de salarios
    print(df['salary_in_usd'].describe())

    # Identificación de tendencias

    # Top 10 trabajos mejor pagados
    top_paying_jobs = df.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False).head(10)
    print("Top 10 trabajos mejor pagados:")
    print(top_paying_jobs)

    # Salario promedio por tamaño de empresa
    salary_by_company_size = df.groupby('company_size')['salary_in_usd'].mean().sort_values(ascending=False)
    print("\nSalario promedio por tamaño de empresa:")
    print(salary_by_company_size)

    # Salario promedio por ratio de trabajo remoto
    remote_work_impact = df.groupby('remote_ratio')['salary_in_usd'].mean().sort_values(ascending=False)
    print("\nSalario promedio por ratio de trabajo remoto:")
    print(remote_work_impact)

    # Tendencias notables:
    # 1. Los trabajos mejor pagados tienden a ser roles de liderazgo o altamente especializados.
    # 2. Las empresas grandes generalmente ofrecen salarios más altos.
    # 3. No hay una clara relación entre el trabajo remoto y los salarios, lo que sugiere
    #    que otros factores como la experiencia y el rol son más determinantes.
    # 4. Hay una gran variabilidad en los salarios, incluso dentro de los mismos niveles de experiencia.

# Ejecutar la función si este archivo es ejecutado directamente
if __name__ == '__main__':
    init_examen()
