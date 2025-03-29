use ndarray::prelude::*;
use price_of_hierarchy::{Cost, Discrete, KMeans, Point, WeightedKMeans, WeightedPoint};
use pyo3::{exceptions::PyValueError, prelude::*};

type PyPoints = Vec<Vec<f64>>;
type PyWeightedPoints = Vec<(f64, Vec<f64>)>;

fn to_points(points: PyPoints) -> Vec<Point> {
    points.into_iter().map(Array1::from_vec).collect()
}

fn to_weighted_points(weighted_points: PyWeightedPoints) -> Vec<WeightedPoint> {
    weighted_points
        .into_iter()
        .map(|(w, v)| (w, Array1::from_vec(v)))
        .collect()
}

#[pyfunction]
fn price_of_kmeans_greedy(points: PyPoints) -> PyResult<f64> {
    Ok(KMeans::new(&to_points(points))
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .price_of_greedy()
        .0)
}

#[pyfunction]
fn price_of_weighted_kmeans_greedy(weighted_points: PyWeightedPoints) -> PyResult<f64> {
    Ok(WeightedKMeans::new(&to_weighted_points(weighted_points))
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .price_of_greedy()
        .0)
}

#[pyfunction]
fn price_of_kmedian_hierarchy(points: PyPoints) -> PyResult<f64> {
    Ok(Discrete::kmedian(&to_points(points))
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .price_of_hierarchy()
        .0)
}

#[pyfunction]
fn price_of_weighted_kmedian_hierarchy(weighted_points: PyWeightedPoints) -> PyResult<f64> {
    Ok(
        Discrete::weighted_kmedian(&to_weighted_points(weighted_points))
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .price_of_hierarchy()
            .0,
    )
}

/// A Python module implemented in Rust.
#[pymodule]
fn clustering_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(price_of_kmeans_greedy, m)?)?;
    m.add_function(wrap_pyfunction!(price_of_weighted_kmeans_greedy, m)?)?;
    m.add_function(wrap_pyfunction!(price_of_kmedian_hierarchy, m)?)?;
    m.add_function(wrap_pyfunction!(price_of_weighted_kmedian_hierarchy, m)?)?;
    Ok(())
}
