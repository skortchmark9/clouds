/* global fetch, setTimeout, clearTimeout */
import React, {useCallback, useEffect, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {Map} from 'react-map-gl/maplibre';
import DeckGL from '@deck.gl/react';
import { CubeGeometry, SphereGeometry } from "@luma.gl/engine";
import {Layer, LightingEffect, _SunLight as SunLight} from '@deck.gl/core';

import {ScenegraphLayer, SimpleMeshLayer} from '@deck.gl/mesh-layers';
import ControlPanel, {
  Button,
  Checkbox,
  Multibox,
  Select,
  Text,
  Color,
  Range,
  Interval,
  Custom,
} from 'react-control-panel';
import type {ScenegraphLayerProps} from '@deck.gl/mesh-layers';
import {type PickingInfo, type MapViewState, OrthographicView} from '@deck.gl/core';
import { CullFaceMode } from 'maplibre-gl';

const sphere = new SphereGeometry();
const cube = new CubeGeometry();

const sun = new SunLight({
  timestamp: 972748800000, // Sat Oct 28 2000 12:00:00 GMT-0400 (Eastern Daylight Time), 
  color: [255, 255, 255],
  intensity: 1
});


const lightingEffect = new LightingEffect({ ambientLight: sun });
const effects = [];
// effects.push(lightingEffect);

const INITIAL_VIEW_STATE: MapViewState = {
  latitude: 38.4940000,
  longitude: -98.4270379,
  zoom: 8,
  pitch: 0,
  minPitch: 0,  
  maxPitch: 90,
  // minPitch: -70,
  // maxPitch: 100,
  maxZoom: 16,
  position: [0, 0, 10_000],
};

const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json';

type Coordinate = {
  lat: number;
  lon: number;
};

type APIResponse = {
  time: string;
  boxes: GridCellAPI[];
};

type GridCellAPI = {
  corners_of_box: [Coordinate, Coordinate, Coordinate, Coordinate];
  total_condensation: number[];
  cell_heights: number[];
};

type GridCell = {
  corner: Coordinate;
  height: number;
  total_condensation: number;
  heightScale: number;
};

function expandAPICells(boxes: GridCellAPI[]): GridCell[] {
  const expanded: GridCell[] = [];
  const HEIGHT_STEP = 0;
  let total = 0;
  let nonNull = 0;
  console.log('api', boxes);
  const heightScales = [55,66,76,85,96,110,131,150,170,190,215,246,277,307,342,382,422,461,481,481,480,480,479,479,478,477,476,476,474,475,474,477,480,484,489,492,496,499,503,509,517,530,542,556,564,578,602,627,609];
  // Expand this so that there's one entity for each height. This
  // is gonna make things slower so maybe don't do this later.
  for (const box of boxes) {
    for (let i = 0; i < box.total_condensation.length; i++) {
      const height = box.cell_heights[i] + HEIGHT_STEP * 20
      const heightScale = heightScales[i];
      // const height = i * HEIGHT_STEP;
      const total_condensation = box.total_condensation[i];

      total++;

      if (total_condensation) {
        const newBox = {
          corner: box.corners_of_box[0],
          total_condensation,
          height,
          heightScale
        }
  
        expanded.push(newBox);
        nonNull++;
        if (nonNull > 2e6) {
          console.log(expanded);
          return expanded;
        }
      }
    }
  }
  console.log(total, nonNull);
  console.log(expanded);
  return expanded;
}

async function fetchGridCells(time, states): Promise<GridCell[]> {
  const url = 'http://127.0.0.1:5000/cloud_data?';
  const params = new URLSearchParams();
  if (time) {
    params.append('time', time);
  }
  for (const state of states) {
    params.append('states', state);
  }
  
  const resp = await fetch(url + params.toString(), { mode: 'cors' })
  // const resp = await fetch('./cloud_mixing_ratio_midwest.json.gz');
  const data = await resp.json() as APIResponse;

  console.log('Got data for time', data.time);
  return expandAPICells(data.boxes);
}

async function fetchEntireUs(): Promise<Record<number, GridCell[]>> {
  const url = '/cloud_condensation_kansas.json.gz';
  const resp = await fetch(url, { mode: 'cors' })
  const data = await resp.json() as APIResponse;
  console.log('Got data for time', data.time);

  const expanded = expandAPICells(data.boxes);
  const output = {};
  for (let i = 0; i < 100; i++) {
    output[i] = expanded;
  }

  return output;
}


async function fetchGridTimeRange(states: string[], bounds: number[]): Promise<Record<number, GridCell[]>> {
  const url = 'http://127.0.0.1:5000/cloud_data_timerange?';

  const output = {};
  const min = bounds[0];
  const max = bounds[1];

  let i = min;
  while(i < max) {
    const params = new URLSearchParams();
    for (const state of states) {
      params.append('states', state);
    }
    // Maybe batches of 10 will not crash as much
    params.append('minTime', (i).toString());
    params.append('maxTime', (i + 10).toString());
    try {
      const resp = await fetch(url + params.toString(), { mode: 'cors' })
      const data = await resp.json() as Record<number, GridCellAPI[]>;  
      for (const [key, value] of Object.entries(data)) {
        output[key] = expandAPICells(value);
      }  
    } catch (e) {
      console.error(e);
      console.log('womp');
    }

    i += 10;
    i = Math.min(max, i);
  }

  return output;
}

export function App({
  sizeScale = 25,
  onDataLoad,
  mapStyle = MAP_STYLE,
  gridcells,
}: {
  sizeScale?: number;
  onDataLoad?: (count: number) => void;
  mapStyle?: string;
  gridcells: GridCell[];
}) {
  const max = (gridcells || []).reduce((acc, d) => Math.max(acc, d.total_condensation), 0) / 2;
  console.log('max', max);
  const translation: [number, number, number] = [2000, 2000, 0];
  const gridcellMesh = new SimpleMeshLayer<GridCell>({
    id: 'SimpleMeshLayer',
    parameters: {
      // blendAlphaOperation: 'max',
      depthMask: false,
      // depthTest: false
    },
    data: gridcells,    
    getColor: (d) => {
      const normalized_cloud = Math.min(d.total_condensation / max);

      const alpha = Math.min(normalized_cloud * 255, 255);
      return [
        100,
        255,
        255,
        alpha,
      ];
    },
    wireframe: false,
    // getOrientation: d => [0, Math.random() * 180, 0],
    getPosition: d => [
      d.corner.lon,
      d.corner.lat,
      d.height,
    ],
    mesh: cube,
    // getOrientation: (d) => [Math.random() * 360, Math.random() * 360, Math.random() * 360],
    getScale: d => [4000, 4000, d.heightScale],
    getTranslation: d => translation,
    pickable: false,
  });

  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);

  // Function to toggle pitch between 0 and 90
  const togglePitch = () => {
    setViewState(prev => ({
      ...prev,
      pitch: prev.pitch === 0 ? 90 : 0
    }));
  };

  const shift = ({left = 0, right = 0, up = 0, down = 0}) => {
    setViewState(prev => ({
      ...prev,
      position: [
        prev.position[0] - left + right,
        prev.position[1] + up - down,
        prev.position[2]
      ]
    }));
  };


  // Listen for the "P" key press to change pitch
  useEffect(() => {
    const handleKeyPress = (event) => {
      if (event.key.toLowerCase() === 'p') {
        togglePitch();
      }
      if (event.key === 'ArrowLeft') {
        shift({left: 4000});
      }
      if (event.key === 'ArrowRight') {
        shift({right: 4000});
      }
      if (event.key === 'ArrowUp') {
        shift({up: 4000});
      }
      if (event.key === 'ArrowDown') {
        shift({down: 4000});
      }

    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [viewState]);


  return (
    <DeckGL
      layers={[gridcellMesh]}
      viewState={viewState}
      onViewStateChange={({ viewState }) => setViewState(viewState)}
      controller={false}
    >
      {/* <Map reuseMaps mapStyle={mapStyle} {...INITIAL_VIEW_STATE} /> */}
    </DeckGL>
  );
}

export default function Wrapper() {
  const [time, setTime] = useState(39);
  const allStates = ['kansas', 'nebraska', 'oklahoma', 'california'];
  const [gridCellsByTime, setGridcellsByTime] =  useState({});
  const [wholeUS, setWholeUS] = useState(true);
  const [currentStates, setCurrentStates] = useState([true, false, false, false]);
  const minTime = 20;
  const maxTime = 80
  const initialState = {
    time: time,
    states: currentStates,
    'Whole US': wholeUS
  };

  useEffect(() => {
    const selectedStates = allStates.filter((_, i) => currentStates[i]);

    if (wholeUS) {
      fetchEntireUs().then((_gridcellsByTime) => {
        setGridcellsByTime(_gridcellsByTime);
      })  
    } else {
      fetchGridTimeRange(selectedStates, [minTime, maxTime]).then((_gridcellsByTime) => {
        setGridcellsByTime(_gridcellsByTime);
      })
    }
  }, [currentStates, wholeUS]);


  const onChange = useCallback((key, value) => {
    if (key === 'time') {
      setTime(value);
    }

    if (key === 'states') {
      setCurrentStates(value.slice());
    }

    if (key === 'Whole US') {
      setWholeUS(value);
    }
  }, [])

  return (<>
    <App gridcells={gridCellsByTime[time]}></App>
    {/* <ControlPanel
      draggable
      theme='dark'
      position={'top-left'}
      initialState={initialState}
      onChange={onChange}
      width={500}
      style={{ marginRight: 30 }}
    >
      <Range label='time' step={1} min={minTime} max={maxTime} />
      <Checkbox label='Whole US'></Checkbox>
      <Multibox
        label='states'
        colors={allStates.map(() => ['rgb(100,120,230)'])}
        names={allStates}
      />
    </ControlPanel> */}
  </>);
}

export function renderToDOM(container: HTMLDivElement) {
  createRoot(container).render(<Wrapper />);
}
