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

// Data provided by the OpenSky Network, http://www.opensky-network.org
// const DATA_URL = 'https://opensky-network.org/api/states/all';
// For local debugging
const DATA_URL = './all.json';
const MODEL_URL = './airplane.glb';
const REFRESH_TIME_SECONDS = 60;
const DROP_IF_OLDER_THAN_SECONDS = 120;


const cube = new CubeGeometry();

const ANIMATIONS: ScenegraphLayerProps['_animations'] = {
  '*': {speed: 1}
};

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


// https://openskynetwork.github.io/opensky-api/rest.html#response
type Aircraft = [
  uniqueId: string,
  callSign: string,
  originCountry: string,
  timePosition: number,
  lastContact: number,
  longitude: number | null,
  latitude: number | null,
  baroAltitude: number | null,
  onGround: boolean,
  velocity: number | null,
  trueAttack: number | null,
  verticalRate: number | null,
  sensors: number[],
  geoAltitude: number | null,
  positionSource: number[],
  category: number
];

const DATA_INDEX = {
  UNIQUE_ID: 0,
  CALL_SIGN: 1,
  ORIGIN_COUNTRY: 2,
  LAST_CONTACT: 4,
  LONGITUDE: 5,
  LATITUDE: 6,
  BARO_ALTITUDE: 7,
  VELOCITY: 9,
  TRUE_TRACK: 10,
  VERTICAL_RATE: 11,
  GEO_ALTITUDE: 13,
  CATEGORY: 17
} as const;

async function fetchData(): Promise<Aircraft[]> {
  const resp = await fetch(DATA_URL);
  const {time, states} = (await resp.json()) as {time: number; states: Aircraft[]};
  // make lastContact timestamp relative to response time
  for (const a of states) {
    a[DATA_INDEX.LAST_CONTACT] -= time;
  }
  return states;
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

function getTooltip({object}: PickingInfo<Aircraft>) {
  return (
    object &&
    `\
    Call Sign: ${object[DATA_INDEX.CALL_SIGN] || ''}
    Country: ${object[DATA_INDEX.ORIGIN_COUNTRY] || ''}
    Vertical Rate: ${object[DATA_INDEX.VERTICAL_RATE] || 0} m/s
    Velocity: ${object[DATA_INDEX.VELOCITY] || 0} m/s
    Direction: ${object[DATA_INDEX.TRUE_TRACK] || 0}`
  );
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
  const [data, setData] = useState<Aircraft[]>();
  const [timer, setTimer] = useState<{id: number | null}>({id: null});

  const [gridcells, setGridcells] = useState<GridCell[]>();

  useEffect(() => {
    timer.id++;
    fetchData()
      .then(newData => {
        if (timer.id === null) {
          // Component has unmounted
          return;
        }
        // In order to keep the animation smooth we need to always return the same
        // object at a given index. This function will discard new objects
        // and only update existing ones.
        if (data) {
          const dataById: Record<string, Aircraft> = {};
          newData.forEach(entry => (dataById[entry[DATA_INDEX.UNIQUE_ID]] = entry));
          newData = data.map(entry => dataById[entry[DATA_INDEX.UNIQUE_ID]] || entry);
        }

      setData(newData);

        if (onDataLoad) {
          onDataLoad(newData.length);
        }
      })
      .finally(() => {
        const timeoutId = setTimeout(() => setTimer({id: timeoutId}), REFRESH_TIME_SECONDS * 1000);
        timer.id = timeoutId;
      });

    return () => {
      clearTimeout(timer.id);
      timer.id = null;
    };
  }, [timer]);

  useEffect(() => {
    fetchGridCells(time, states).then((_gridcells) => {
      setGridcells(_gridcells);
    })
  }, [time, states]);

  const layer = new ScenegraphLayer<Aircraft>({
    id: 'scenegraph-layer',
    data,
    pickable: true,
    sizeScale,
    scenegraph: MODEL_URL,
    _animations: ANIMATIONS,
    sizeMinPixels: 0.1,
    sizeMaxPixels: 1.5,
    getPosition: d => [
      d[DATA_INDEX.LONGITUDE] ?? 0,
      d[DATA_INDEX.LATITUDE] ?? 0,
      d[DATA_INDEX.GEO_ALTITUDE] ?? 0
    ],
    getOrientation: d => {
      const verticalRate = d[DATA_INDEX.VERTICAL_RATE] ?? 0;
      const velocity = d[DATA_INDEX.VELOCITY] ?? 0;
      // -90 looking up, +90 looking down
      const pitch = (-Math.atan2(verticalRate, velocity) * 180) / Math.PI;
      const yaw = -d[DATA_INDEX.TRUE_TRACK] ?? 0;
      return [pitch, yaw, 90];
    },
    getScale: d => {
      const lastContact = d[DATA_INDEX.LAST_CONTACT];
      return lastContact < -DROP_IF_OLDER_THAN_SECONDS ? [0, 0, 0] : [1, 1, 1];
    },
    transitions: {
      getPosition: REFRESH_TIME_SECONDS * 1000
    }
  });

  const HEIGHT = 10;
  const gridcellScene = new ScenegraphLayer<GridCell>({
    id: 'gridcell-layer',
    data: gridcells,
    pickable: true,
    sizeScale,
    scenegraph: MODEL_URL,
    sizeMinPixels: 0.1,
    sizeMaxPixels: 1.5,
    getPosition: d => [
      d.corner.lon,
      d.corner.lat,
      HEIGHT,
    ],
    transitions: {
      getPosition: REFRESH_TIME_SECONDS * 1000
    }
  });

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
      getTooltip={getTooltip}
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
  </>)
  return <App time={time}></App>
}

export function renderToDOM(container: HTMLDivElement) {
  createRoot(container).render(<Wrapper />);
}
