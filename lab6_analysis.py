import numpy as np
years_exp = np.array([1, 3, 5, 7, 10])
print("Years of Experience:", years_exp)
salaries = np.array([[50, 60, 70], [80, 90, 100]])
print("Salary Matrix:\n", salaries)
zeros_array = np.zeros((2, 2)) # 2x2 zeros
ones_array = np.ones((2, 3)) # 2x3 ones
print("Zeros Array:\n", zeros_array)
print("Ones Array:\n", ones_array)
years_exp = np.append(years_exp, [12, 15])
new_salaries = np.array([[110, 120, 130]])
salaries = np.append(salaries, new_salaries, axis=0)
identity_mtx = np.eye(3)
print("Updated Years of Experience:", years_exp)
print("\nUpdated Salary Matrix:\n", salaries)
print("\n3x3 Identity Matrix:\n", identity_mtx)
exp_plus_5 = years_exp + 5 # Add 5 years to all experience values
print("Years + 5:", exp_plus_5)
exp_times_2 = years_exp * 2 # Multiply all values by 2
print("Years * 2:", exp_times_2)
sample1 = np.array([1, 2, 3])
sample2 = np.array([4, 5, 6])
dot_result = np.dot(sample1, sample2)
print("Dot Product:", dot_result)
exp_minus_2 = years_exp - 2
exp_div_2 = years_exp / 2
exp_values = np.exp(years_exp)
log_values = np.log(years_exp)
print("Subtraction Result:", exp_minus_2)
print("Logarithm Values:", log_values)
print("First year of experience:", years_exp[0])
print("First two salaries:", salaries[0, :2])
print("Second column salaries:", salaries[:, 1])
print("Last year of experience:", years_exp[-1])
reversed_exp = years_exp[::-1]
salary_subgroup = salaries[0:2, 1:3]
print("Reversed Experience:", reversed_exp)
print("Salary Subgroup (2x2 block):\n", salary_subgroup)
reshaped_exp = np.reshape(np.arange(1, 7), (2, 3))
print("Reshaped Experience Array:\n", reshaped_exp)
flattened_exp = reshaped_exp.flatten()
print("Flattened Array:", flattened_exp)
print("Transposed Array:\n", reshaped_exp.T)
data = np.arange(1, 13)
matrix_3x4 = data.reshape(3, 4)
tensor_3d = data.reshape(2, 2, 3)
print("3x4 Matrix:\n", matrix_3x4)
print("\n3D Tensor:\n", tensor_3d)
bonus = np.array([5, 10, 15])
salaries_with_bonus = salaries + bonus
print("Salaries after bonus:\n", salaries_with_bonus)
scaling_factor = 1.10
scaled_salaries = salaries * scaling_factor
column_scalars = np.array([1.1, 1.2, 1.3])
specific_scaled = salaries * column_scalars
print("Salaries with 10% raise:\n", scaled_salaries)
print("Mean experience:", np.mean(years_exp))
print("Std deviation of experience:", np.std(years_exp))
print("Max salary:", np.max(salaries), "Min salary:", np.min(salaries))
print("Sum of all salaries:", np.sum(salaries))
median_exp = np.median(years_exp)
p75_salaries = np.percentile(salaries, 75)
print(f"Median Experience: {median_exp}")
print(f"75th Percentile of Salaries: {p75_salaries}")
angles = np.array([0, np.pi/4, np.pi/2])
print("Sine of angles:", np.sin(angles))
print("Cosine of angles:", np.cos(angles))
salary_sums = np.apply_along_axis(np.sum, 1, salaries)
print("Sum of Salaries per person:", salary_sums)
sqrt_salaries = np.sqrt(salaries)
print("Square root of salaries:\n", sqrt_salaries)
def get_range(row):
    return np.max(row) - np.min(row)
salary_ranges = np.apply_along_axis(get_range, 1, salaries)
print("Salary range per person:", salary_ranges)
import pandas as pd
data = np.random.randint(1, 50, size=(5, 3))
df_data = pd.DataFrame(data, columns=['X', 'Y', 'Z'])
print("Generated Data:\n", df_data)
df_data['Log_X'] = np.log(df_data['X'])
df_data['Sqrt_Y'] = np.sqrt(df_data['Y'])
print("DataFrame with NumPy Transformations:\n", df_data)
print("Correlation Matrix:\n", df_data.corr())
print("Mean of Column X:", np.mean(df_data['X']))
print("Median of Column Y:", np.median(df_data['Y']))
df_data['Squared_Z'] = np.square(df_data['Z'])
print("\nDataFrame with Squared Z:\n", df_data)
df_data.to_csv('sample_data.csv', index=False)
print("Data saved to 'sample_data.csv'.")
df_imported = pd.read_csv('sample_data.csv')
print("Imported DataFrame:\n", df_imported)
summary_stats = df_imported.describe()
print("Summary Statistics:\n", summary_stats)
print("Column Means:\n", df_imported.mean())
print("Column Standard Deviations:\n", df_imported.std())
df_imported['Sum_XY'] = df_imported['X'] + df_imported['Y']
df_imported.to_csv('final_data_analysis.csv', index=False)
print("\nFinal Modified DataFrame:\n", df_imported.head())
print("\nSuccess! Modified data saved to 'final_data_analysis.csv'.")
df_kaggle = pd.read_csv('survey_results_public.csv')
print("Loaded Dataset:\n", df_kaggle.head())
df_subset = df_kaggle[['Country', 'EdLevel', 'YearsCodePro', 'ConvertedCompYearly']]
print("Subset of Data:\n", df_subset.head())
df_clean = df_subset.dropna()
print("Cleaned Data:\n", df_clean.head())
years_numeric = pd.to_numeric(df_clean['YearsCodePro'], errors='coerce')
df_clean['ExperienceLevel'] = np.where(years_numeric >= 10, 'Senior', 'Junior')
grouped_data = df_clean.groupby(['Country', 'ExperienceLevel'])['ConvertedCompYearly'].mean()
print("Formatted Grouped Data:\n", grouped_data.head())
grouped_data = grouped_data.reset_index()
print("Formatted Grouped Data:\n", grouped_data.head())
edu_stats = df_clean.groupby('EdLevel')['ConvertedCompYearly'].agg(['median', 'std'])
print("Salary Stats by Education Level:\n", edu_stats)
country_comp = df_clean.groupby('Country')['ConvertedCompYearly'].mean()
top_10_countries = country_comp.sort_values(ascending=False).head(10)
print("Top 10 Highest Paying Countries:\n", top_10_countries)
# List of languages provided in your exercise
languages = ['C#', 'Flutter', 'Java', 'JavaScript', 'Matlab', 'PhP', 'Python', 'React', 'Swift', 'TypeScript']
languages.sort()
student_last_three = 585
lang_index = student_last_three % 10
assigned_lang = languages[lang_index]
print(f"Your assigned language is: {assigned_lang}")
import pandas as pd
import numpy as np
df = pd.read_csv('Most Popular Programming Languages.csv')
print("Before conversion:\n", df['2026-04'].head())

