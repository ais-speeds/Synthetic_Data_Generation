import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sdmetrics.reports.single_table import DiagnosticReport, QualityReport
from sdmetrics.visualization import get_column_plot, get_column_pair_plot

from sdmetrics.single_column import KSComplement, TVComplement, RangeCoverage, CategoryCoverage, MissingValueSimilarity, StatisticSimilarity
from sdmetrics.column_pairs import CorrelationSimilarity
from sdmetrics.single_table import NewRowSynthesis

class SyntheticDataEvaluator:
    def __init__(self, real_data, synthetic_data, metadata):
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.metadata = metadata.to_dict()
        
        # Identify numerical and categorical columns
        self.numerical_columns = [
            col_name for col_name, col_value in self.metadata['columns'].items()
            if col_value['sdtype'] == 'numerical'
        ]
        self.categorical_columns = [
            col_name for col_name, col_value in self.metadata['columns'].items()
            if col_value['sdtype'] == 'categorical'
        ]
        
        
    def ks_complement_eval(self):
        """Evaluate KSComplement for all numerical columns."""
        
        ks_scores = {}
        for column_name in self.numerical_columns:
            column_score = KSComplement.compute(self.real_data[column_name], self.synthetic_data[column_name])
            ks_scores[column_name] = column_score
        
        # Convert to dataframe for better readability
        ks_scores_df = pd.DataFrame(ks_scores.items(), columns=['Variable', 'KSComplement Score'])
        
        # Sort the dataframe by 'KSComplement Score' in descending order
        ks_scores_df = ks_scores_df.sort_values('KSComplement Score', ascending=False)
        
        # Compute mean KSComplement score and append it to the dataframe
        mean_ks_score = sum(ks_scores.values()) / len(ks_scores)
        mean_ks_row = pd.DataFrame([['mean_ks_score', mean_ks_score]], columns=['Variable', 'KSComplement Score'])
        ks_scores_df = pd.concat([ks_scores_df, mean_ks_row], ignore_index=True)
        
        return ks_scores_df
    
    def tv_complement_eval(self):
        """Evaluate TVComplement for all categorical columns."""
        
        tv_scores = {}
        for column_name in self.categorical_columns:
            column_score = TVComplement.compute(self.real_data[column_name], self.synthetic_data[column_name])
            tv_scores[column_name] = column_score
        
        # Convert to dataframe for better readability
        tv_scores_df = pd.DataFrame(tv_scores.items(), columns=['Variable', 'TVComplement Score'])
        
        # Sort the dataframe by 'TVComplement Score' in descending order
        tv_scores_df = tv_scores_df.sort_values('TVComplement Score', ascending=False)
        
        # Compute mean TVComplement score and append it to the dataframe
        mean_tv_score = sum(tv_scores.values()) / len(tv_scores)
        mean_tv_row = pd.DataFrame([['mean_tv_score', mean_tv_score]], columns=['Variable', 'TVComplement Score'])
        tv_scores_df = pd.concat([tv_scores_df, mean_tv_row], ignore_index=True)
        
        return tv_scores_df

    def descr_stat_similarity_eval(self):
        """Evaluate descriptive statistics (mean, median, std) for all numerical columns."""
        
        stats = ['mean', 'median', 'std']
        results = {}
        
        for column_name in self.numerical_columns:
            column_results = {}
            for stat in stats:
                score = StatisticSimilarity.compute(
                    self.real_data[column_name],
                    self.synthetic_data[column_name],
                    statistic=stat
                    )
                column_results[stat] = score
            results[column_name] = column_results
            
        # Flatten the nested dictionaries into a dataframe
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df.reset_index(inplace=True)
        results_df.rename(columns={'index': 'Variable'}, inplace=True)
            
        return results_df
    
    def corr_similarity_eval(self):
        """Evaluate correlation similarity for all pairs of numerical columns."""
        
        # Prompt user to choose the correlation coefficient
        print("Choose correlation coefficient:")
        print("1. Pearson")
        print("2. Spearman")
        choice = int(input("Enter choice (1 or 2): "))
        
        if choice == 1:
            coefficient = 'Pearson'
        elif choice == 2:
            coefficient = 'Spearman'
        else:
            raise ValueError("Invalid choice. Please enter 1 for Pearson or 2 for Spearman.")

        correlation_scores = {}
        total_score = 0
        total_pairs = 0
        
        for col_A, col_B in itertools.combinations(self.numerical_columns, 2):
            score = CorrelationSimilarity.compute(
                real_data=self.real_data[[col_A, col_B]],
                synthetic_data=self.synthetic_data[[col_A, col_B]],
                coefficient=coefficient
            )
            
            correlation_scores[(col_A, col_B)] = score
            total_score += score
            total_pairs += 1
        
        # Convert to dataframe for better readability
        correlation_scores_df = pd.DataFrame.from_dict(correlation_scores, orient='index', columns=['Correlation Similarity'])
        correlation_scores_df.index.name = 'Column Pair'
        correlation_scores_df.reset_index(inplace=True)
        
        # Sort the dataframe by 'Correlation Similarity' in descending order
        correlation_scores_df = correlation_scores_df.sort_values('Correlation Similarity', ascending=False)
        
        # Compute mean correlation similarity score and append it to the dataframe
        mean_correlation_score = total_score / total_pairs if total_pairs > 0 else None
        mean_correlation_row = pd.DataFrame([['mean_correlation_score', mean_correlation_score]], columns=['Column Pair', 'Correlation Similarity'])
        correlation_scores_df = pd.concat([correlation_scores_df, mean_correlation_row], ignore_index=True)
        
        return correlation_scores_df
        
    def range_coverage_eval(self):
        """Evaluate RangeCoverage for all numerical columns."""
                
        range_coverage_scores = {}
        
        for column_name in self.numerical_columns:
            column_score = RangeCoverage.compute(self.real_data[column_name], self.synthetic_data[column_name])
            range_coverage_scores[column_name] = column_score
        
        # Convert to dataframe
        range_coverage_df = pd.DataFrame.from_dict(range_coverage_scores, orient='index', columns=['Range Coverage'])
        range_coverage_df.index.name = 'Variable'
        range_coverage_df.reset_index(inplace=True)
        
        # Sort the dataframe by 'Range Coverage' in descending order
        range_coverage_df = range_coverage_df.sort_values('Range Coverage', ascending=False)
        
        # Compute mean range coverage score and append it to the dataframe
        mean_range_coverage = sum(range_coverage_scores.values()) / len(range_coverage_scores)
        mean_range_coverage_row = pd.DataFrame([['mean_range_coverage', mean_range_coverage]], columns=['Variable', 'Range Coverage'])
        range_coverage_df = pd.concat([range_coverage_df, mean_range_coverage_row], ignore_index=True)
        
        return range_coverage_df
    
    def cat_coverage_eval(self):
        """Evaluate CategoryCoverage for all categorical columns."""
        
        cat_coverage_scores = {}
        
        for column_name in self.categorical_columns:
            column_score = CategoryCoverage.compute(self.real_data[column_name], self.synthetic_data[column_name])
            cat_coverage_scores[column_name] = column_score
        
        # Convert to dataframe
        cat_coverage_df = pd.DataFrame.from_dict(cat_coverage_scores, orient='index', columns=['Category Coverage'])
        cat_coverage_df.index.name = 'Variable'
        cat_coverage_df.reset_index(inplace=True)
        
        # Sort the dataframe by 'Category Coverage' in descending order
        cat_coverage_df = cat_coverage_df.sort_values('Category Coverage', ascending=False)
        
        # Compute mean category coverage score and append it to the dataframe
        mean_cat_coverage = sum(cat_coverage_scores.values()) / len(cat_coverage_scores)
        mean_cat_coverage_row = pd.DataFrame([['mean_category_coverage', mean_cat_coverage]], columns=['Variable', 'Category Coverage'])
        cat_coverage_df = pd.concat([cat_coverage_df, mean_cat_coverage_row], ignore_index=True)   

        return cat_coverage_df
        
    def miss_val_similarity_eval(self):
        """Evaluate MissingValueSimilarity for columns with missing values."""
        
        missing_val_scores = {}
        total_missing_val_score = 0
        columns_with_missing_values = 0

        for column in self.real_data.columns:
            if self.real_data[column].isna().any() or self.synthetic_data[column].isna().any():
                columns_with_missing_values += 1
                score = MissingValueSimilarity.compute(self.real_data[column], self.synthetic_data[column])
                missing_val_scores[column] = score
                total_missing_val_score += score

        if columns_with_missing_values > 0:
            mean_missing_val_score = total_missing_val_score / columns_with_missing_values
        else:
            mean_missing_val_score = None  # Indicates that no columns have missing values

        missing_val_scores['mean_missing_value_similarity'] = mean_missing_val_score
        
        # Convert to dataframe
        missing_val_scores_df = pd.DataFrame.from_dict(missing_val_scores, orient='index', columns=['Missing Value Similarity'])
        missing_val_scores_df.index.name = 'Variable'
        missing_val_scores_df.reset_index(inplace=True)

        return missing_val_scores_df
    
    def new_row_synthesis_eval(self, numerical_match_tolerance=0.01, synthetic_sample_size=None):
        """Evaluate New Row Synthesis for the synthetic data."""

        # Set default synthetic sample size if not provided
        if synthetic_sample_size is None:
            synthetic_sample_size = self.real_data.shape[0]

        score = NewRowSynthesis.compute(
            real_data=self.real_data,
            synthetic_data=self.synthetic_data,
            metadata=self.metadata,
            numerical_match_tolerance=numerical_match_tolerance,
            synthetic_sample_size=synthetic_sample_size
        )
        
        # Convert to dataframe
        new_row_synthesis_df = pd.DataFrame({'New Row Synthesis': [score]})

        return new_row_synthesis_df
    
    def generate_diagnostic_report(self, visualize=True):
        """Generate and return the Diagnostic Report with user input for diagnostic type."""
        
        # Available diagnostic types
        diagnostic_types = ['Data Validity', 'Data Structure']
        
        # Prompt user to choose the detail type
        print("Choose diagnostic type for Diagnostic Report:")
        for i, option in enumerate(diagnostic_types, 1):
            print(f"{i}. {option}")
        choice = int(input("Enter choice (1 or 2): ")) - 1
        diagnostic_type = diagnostic_types[choice]
        
        # Generate the report
        diagnostic = DiagnosticReport()
        diagnostic.generate(self.real_data, self.synthetic_data, self.metadata)

        details = diagnostic.get_details(diagnostic_type)
    
        if visualize:
            diagnostic.get_visualization(diagnostic_type)
    
        return details
        
    def generate_quality_report(self, visualize=True):
        """Generate and return the Quality Report with user input for detail type."""
        # Available detail types
        detail_types = ['Column Shapes', 'Column Pair Trends']
        
        # Prompt user to choose the detail type
        print("Choose detail type for Quality Report:")
        for i, option in enumerate(detail_types, 1):
            print(f"{i}. {option}")
        choice = int(input("Enter choice (1 or 2): ")) - 1
        detail_type = detail_types[choice]
        
        # Generate the report
        quality_report = QualityReport()
        quality_report.generate(self.real_data, self.synthetic_data, self.metadata)
        
        details = quality_report.get_details(detail_type)
    
        if visualize:
            quality_report.get_visualization(detail_type)
        
        return details

    def plot_columns(self):
        """Visualize columns based on user choice of numerical or categorical columns."""
        
        column_types = [self.numerical_columns, self.categorical_columns]

        # Prompt user to choose the column type
        print("Choose column type for visualization:")
        for i, option in enumerate(['Numerical', 'Categorical'], 1):
            print(f"{i}. {option}")
        choice = int(input("Enter choice (1 or 2): ")) - 1
        columns = column_types[choice]
        
        # Choose columns based on user input
        if choice == 0:
            columns = self.numerical_columns
            plot_type = 'distplot'
        else:
            columns = self.categorical_columns
            plot_type = 'bar'
            
        # Determine the grid shape for the subplots
        rows = cols = int(len(columns) ** 0.5) + 1
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=columns)
        
        for i, column_name in enumerate(columns, 1):
            # Create each column plot using SDV's 'get_column_plot'
            column_fig = get_column_plot(
                real_data=self.real_data,
                synthetic_data=self.synthetic_data,
                column_name=column_name,
                plot_type=plot_type
            )
            
            # Determine subplot position
            row = (i - 1) // cols + 1
            col = (i - 1) % cols + 1
            
            # Add trace to the corresponding subplot
            for trace in column_fig['data']:
                fig.add_trace(trace, row=row, col=col)
                
        
        # Update layout and show the figure
        fig.update_layout(height=300*rows, width=300*cols, title_text="Real vs Synthetic Data")
        fig.show()
    
    def plot_column_pairs(self, column_names, plot_type='scatter'):
        """Visualize the top 4 column pairs with highest correlation similarity."""
        
        # Calculate correlation similarity scores
        correlation_scores = self.corr_similarity_eval()

        # Exclude the mean score and sort pairs by their correlation similarity
        sorted_pairs = sorted(correlation_scores.items(), key=lambda x: x[1], reverse=True)
        top_pairs = [pair for pair, score in sorted_pairs if pair != 'mean_correlation_score'][:4]

        # Create a 2x2 grid layout for the subplots
        fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{pair[0]} vs {pair[1]}" for pair in top_pairs])

        for i, pair in enumerate(top_pairs, 1):
            # Generate the column pair plot using SDV's 'get_column_pair_plot'
            column_pair_fig = get_column_pair_plot(
                real_data=self.real_data,
                synthetic_data=self.synthetic_data,
                column_names=list(pair),
                plot_type='scatter'
            )

        # Determine subplot position
        row = (i - 1) // 2 + 1
        col = (i - 1) % 2 + 1

        # Add trace to the corresponding subplot
        for trace in column_pair_fig['data']:
            fig.add_trace(trace, row=row, col=col)

        # Update layout and show the figure
        fig.update_layout(height=600, width=600, title_text="Real vs Synthetic Column Pair Correlations")
        fig.show()
        
    def apply_all_metrics(self):
        """Apply all metrics and reports in this class to evaluate your synthetic data."""

        # Apply individual metrics
        ks_scores = self.ks_complement_eval()
        tv_scores = self.tv_complement_eval()
        descr_stats = self.descr_stat_similarity_eval()
        corr_scores = self.corr_similarity_eval()
        range_coverage_scores = self.range_coverage_eval()
        category_coverage_scores = self.cat_coverage_eval()
        missing_value_scores = self.miss_val_similarity_eval()
        new_row_synthesis_score = self.new_row_synthesis_eval()

        # Generate reports
        diagnostic_report_details = self.generate_diagnostic_report(visualize=True)
        quality_report_details = self.generate_quality_report(visualize=True)

        # Compile results
        results = {
            'KSComplement': ks_scores,
            'TVComplement': tv_scores,
            'DescriptiveStatsSimilarity': descr_stats,
            'CorrelationSimilarity': corr_scores,
            'RangeCoverage': range_coverage_scores,
            'CategoryCoverage': category_coverage_scores,
            'MissingValueSimilarity': missing_value_scores,
            'NewRowSynthesis': new_row_synthesis_score,
            'DiagnosticReport': diagnostic_report_details,
            'QualityReport': quality_report_details
        }

        return results