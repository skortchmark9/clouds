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
  maxZoom: 16,
  pitch: 0,
  bearing: 0
};

const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json';

type Coordinate = {
  lat: number;
  lon: number;
};

type GridCellAPI = {
  corners_of_box: [Coordinate, Coordinate, Coordinate, Coordinate];
  cloud_mixing_ratios: number[];
  ice_mixing_ratios: number[];
};

type GridCell = {
  corner: Coordinate;
  height: number;
  cloud_mixing_ratio: number;
  ice_mixing_ratio: number;
};

type GridCellsAtTime = {
  time: number;
  gridcells: GridCell[];
};


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

  const expanded: GridCell[] = [];
  const HEIGHT_SCALE = 15_000;


  let total = 0;
  let nonNull = 0;
  // Expand this so that there's one entity for each height. This
  // is gonna make things slower so maybe don't do this later.
  for (const box of data) {
    for (let i = 0; i < box.cloud_mixing_ratios.length; i++) {
      const height = i * HEIGHT_SCALE;
      const cloud_mixing_ratio = box.cloud_mixing_ratios[i];
      const ice_mixing_ratio = box.ice_mixing_ratios[i];

      total++;

      if (cloud_mixing_ratio || ice_mixing_ratio) {
        const newBox = {
          corner: box.corners_of_box[0],
          cloud_mixing_ratio,
          ice_mixing_ratio,
          height,
        }

        // if (expanded.length > 1e6) {
        //   continue;
        // }
  
        expanded.push(newBox);
        nonNull++;
      }
    }
  }

  console.log(`${Math.round(100 * nonNull / total)}% of cells have nonzero cloud mixing ratio`)
  console.log(expanded);
  return expanded;
}

export function App({
  sizeScale = 25,
  onDataLoad,
  mapStyle = MAP_STYLE,
  time,
  states
}: {
  sizeScale?: number;
  onDataLoad?: (count: number) => void;
  mapStyle?: string;
  time: number;
  states: string[];
}) {
  const [gridcells, setGridcells] = useState<GridCell[]>();

  useEffect(() => {
    fetchGridCells(time, states).then((_gridcells) => {
      setGridcells(_gridcells);
    })
  }, [time, states]);


  const gridcellMesh = new SimpleMeshLayer<GridCell>({
    id: 'SimpleMeshLayer',
    data: gridcells,    
    getColor: (d) => {
      const max_cloud_mixing = 0.0017;
      const max_ice_mixing = 0.0013;

      const normalized_cloud = d.cloud_mixing_ratio / max_cloud_mixing;
      const normalized_ice = d.ice_mixing_ratio / max_ice_mixing;

      const alpha = (normalized_ice + normalized_cloud + 0.2) * 255;
      const red = normalized_cloud * 255;
      const blue = normalized_ice * 255;
      return [red, blue, 140, alpha]
    },
    // getOrientation: d => [0, Math.random() * 180, 0],
    getPosition: d => [
      d.corner.lon,
      d.corner.lat,
      d.height,
    ],
    mesh: cube,
    sizeScale: 3000,
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
  const allStates = ['kansas', 'nebraska', 'oklahoma'];
  const [currentStates, setCurrentStates] = useState([true, false, false]);
  const initialState = {
    time: time,
    states: currentStates
  };

  const onChange = useCallback((key, value) => {
    if (key === 'time') {
      setTime(value);
    }

    if (key === 'states') {
      setCurrentStates(value.slice());
    }
  }, [])

  const selectedStates = allStates.filter((_, i) => currentStates[i]);
  console.log(selectedStates);

  return (<>
    <App time={time} states={selectedStates}></App>
    <ControlPanel
      draggable
      theme='dark'
      position={'top-left'}
      initialState={initialState}
      onChange={onChange}
      width={500}
      style={{ marginRight: 30 }}
    >
      <Range label='time' min={0} max={100} />
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
