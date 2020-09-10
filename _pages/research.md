---
layout: archive
title: "Research"
permalink: /research/
author_profile: true
---
I have been working on studying the effects of environment on properties of galaxies. My work can be categorized in following disciplines:

- Analytical models to understand importance of various processes at work in large-scale environments like clusters of galaxies.
- Using multi-wavelength data to study galaxies in large-scale environments.
- Comparing results from large-scale high resolution simulations like EAGLE, Illustris and Millennium simulations with observations.
- Hydrodynamical simulation using RAMSES code to study impact of quenching mechanisms in galactic encounters. 


{% include base_path %}

{% for post in site.code reversed %}  
  {% if post.collection == 'research' %}  
    {% include archive-single.html %}  
  {% endif %}  
{% endfor %}  