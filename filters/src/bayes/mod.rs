//! Discrete Bayes Filter
//!
//! This is licensed under an MIT license. See the readme.MD file
//! for more information.

use ndarray::prelude::*;
use ndarray::{ArrayD, ScalarOperand};
use num_traits::Zero;
use ocl_convolution::{Convolution, FeatureMap, Params};
use std::error::Error;
use std::ops::{Div, Mul};

#[derive(PartialEq)]
pub enum PDMode {
    Wrap,
    Constant,
}

pub struct PredictionConfig {
    offset: u8,
    mode: PDMode,
    cval: f32,
}

impl Default for PredictionConfig {
    fn default() -> Self {
        PredictionConfig {
            offset: 0,
            mode: PDMode::Wrap,
            cval: 0.0,
        }
    }
}

fn roll<P, T>(a: &mut ArrayD<P>, offset: T) -> ArrayD<P>
where
    P: Clone,
    T: Into<isize>,
{
    let offset: isize = offset.into();

    let mut uninit = Vec::with_capacity(a.len());
    unsafe {
        uninit.set_len(a.len());
    }

    let mut b = Array::from_vec(uninit).into_shape(a.dim()).unwrap();
    b.slice_mut(s![.., offset..])
        .assign(&a.slice(s![.., ..-offset]));
    b.slice_mut(s![.., ..offset])
        .assign(&a.slice(s![.., -offset..]));
    b
}

fn shift<P, T>(a: &mut ArrayD<P>, offset: T, cval: P) -> ArrayD<P>
where
    P: Clone,
    T: Into<isize>,
{
    let offset: isize = offset.into();

    let mut uninit = Vec::with_capacity(a.len());
    unsafe {
        uninit.set_len(a.len());
    }

    let mut b = Array::from_vec(uninit).into_shape(a.dim()).unwrap();
    b.slice_mut(s![.., offset..]).map(|x| *x = cval);
    b.slice_mut(s![.., ..offset])
        .assign(&a.slice(s![.., -offset..]));
    b
}

pub fn normalize<P>(pdf: ArrayD<P>) -> ArrayD<P>
where
    P: ScalarOperand + Zero + Div<Output = P>,
{
    let sum = pdf.sum();
    pdf / sum
}

pub fn update<P>(likelihood: P, mut prior: ArrayD<P>)
where
    P: ScalarOperand + Zero + Mul<Output = P> + Div<Output = P>,
{
    let raw_update = prior * likelihood;
    prior = normalize(raw_update)
}

pub fn predict(
    mut pdf: ArrayD<f32>,
    kernel: ArrayD<f32>,
    pred_conf: PredictionConfig,
) -> Result<ArrayD<f32>, Box<dyn Error>> {
    let convolution = Convolution::f32(kernel.ndim())? // TODO: Check if it works
        .build(Params {
            strides: [1, 1],
            pads: [0; 0],
            dilation: [1, 1],
            groups: 1,
        })?
        .with_filters(&kernel)?;

    if pred_conf.mode == PDMode::Wrap {
        return convolution.compute(FeatureMap::nhwc(&roll(&mut pdf, pred_conf.offset)));
    }

    convolution.compute(FeatureMap::nhwc(&shift(
        &mut pdf,
        pred_conf.offset,
        pred_conf.cval,
    )))
}
