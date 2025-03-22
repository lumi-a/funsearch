use ndarray::prelude::*;
use price_of_hierarchy::{Cost, Discrete, KMeans};
use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn price_of_kmeans_greedy(points: Vec<Vec<f64>>) -> PyResult<f64> {
    let points: Vec<Array1<f64>> = points.into_iter().map(Array1::from_vec).collect();
    let ratio = KMeans::new(&points).unwrap().price_of_greedy().0;
    Ok(ratio)
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn price_of_kmedian_hierarchy(points: Vec<Vec<f64>>) -> PyResult<f64> {
    let points: Vec<Array1<f64>> = points.into_iter().map(Array1::from_vec).collect();
    let ratio = Discrete::kmedian(&points).unwrap().price_of_hierarchy().0;
    Ok(ratio)
}

/// A Python module implemented in Rust.
#[pymodule]
fn clustering_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(price_of_kmeans_greedy, m)?)?;
    m.add_function(wrap_pyfunction!(price_of_kmedian_hierarchy, m)?)?;
    Ok(())
}
