/* global fetch, setTimeout, clearTimeout */
import React, {useCallback, useEffect, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {Map} from 'react-map-gl/maplibre';
import DeckGL from '@deck.gl/react';
import { CubeGeometry } from "@luma.gl/engine";
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
import type {PickingInfo, MapViewState} from '@deck.gl/core';


const cube = new CubeGeometry();

const INITIAL_VIEW_STATE: MapViewState = {
  latitude: 39.1,
  longitude: -94.57,
  zoom: 3.8,
  minPitch: -70,
  maxZoom: 16,
  pitch: 90,
  bearing: 0
};

const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json';

type Coordinate = {
  lat: number;
  lon: number;
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
};

function expandAPICells(boxes: GridCellAPI[]): GridCell[] {
  const expanded: GridCell[] = [];
  const HEIGHT_STEP = 4000;
  let total = 0;
  let nonNull = 0;
  // Expand this so that there's one entity for each height. This
  // is gonna make things slower so maybe don't do this later.
  for (const box of boxes) {
    for (let i = 0; i < box.total_condensation.length; i++) {
      const height = i * HEIGHT_STEP;
      const total_condensation = box.total_condensation[i];

      total++;

      if (total_condensation) {
        const newBox = {
          corner: box.corners_of_box[0],
          total_condensation,
          height,
        }
  
        expanded.push(newBox);
        nonNull++;
        // if (nonNull > 2e6) {
        //   console.log(expanded);
        //   return expanded;
        // }
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
  const data = await resp.json() as GridCellAPI[];

  return expandAPICells(data);
}

async function fetchEntireUs(): Promise<Record<number, GridCell[]>> {
  const url = '/cloud_condensation_kansas.json.gz';
  const resp = await fetch(url, { mode: 'cors' })
  const data = await resp.json() as GridCellAPI[];

  const expanded = expandAPICells(data);
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
  const gridcellMesh = new SimpleMeshLayer<GridCell>({
    id: 'SimpleMeshLayer',
    data: gridcells,    
    getColor: (d) => {
      const max = 0.002;
      const normalized_cloud = d.total_condensation / max;

      const alpha = Math.min(normalized_cloud * 255, 255);
      // const alpha = 255;
      return [120, 255, 140, alpha]
    },
    // getOrientation: d => [0, Math.random() * 180, 0],
    getPosition: d => [
      d.corner.lon,
      d.corner.lat,
      d.height,
    ],
    mesh: cube,
    sizeScale: 2000,
    pickable: false,
  });

  return (
    <DeckGL
      layers={[gridcellMesh]}
      initialViewState={INITIAL_VIEW_STATE}
      controller={true}
    >
      <Map reuseMaps mapStyle={mapStyle} />
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
    <ControlPanel
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
    </ControlPanel>
  </>);
}

export function renderToDOM(container: HTMLDivElement) {
  createRoot(container).render(<Wrapper />);
}
