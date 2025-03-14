#!/bin/bash

#############
# Servicio Meteorologico Nacional
# Autores: Maximiliano A. Sacco; Yanina Garcia Skabar; Maria Eugenia Dillon; Cynthia Mariana Matsudo
# Fecha: 03/2019
# Adaptado al cluster hydra en CIMA
# Fecha: 06/2023
#############

###  Leyenda de USO

read -r -d '' USO << EOF

Use: createExpFCST.sh  "TEMPLATE_NAME" "BASEDIR"

EOF
: ${2?"$USO"}

WRFUTILDIR=$(pwd)/../

### PARAMETROS

export TEMPLATE_NAME=$1    #Which template will be used to create the experiment (see folders in /conf )
export BASEDIR=$2          #Which is the root directory of the experiment.

source $WRFUTILDIR/lib/errores.env
##### FIN INICIALIZACION ######

EXPDIR=$BASEDIR
[ -d "$BASEDIR" ]  && dispararError 6 "$BASEDIR"
mkdir -p $BASEDIR || dispararError 5 "$BASEDIR"


### Copio la configuracion y genero los archivos de configuracion para el experimento.
mkdir $BASEDIR/conf
cp $WRFUTILDIR/conf/${TEMPLATE_NAME}/* $BASEDIR/conf/
sed -i -e "s|__BASEDIR__|$BASEDIR|g"   $BASEDIR/conf/config.env
source $BASEDIR/conf/config.env


###  Creando Entorno 
mkdir -p $LOGDIR
mkdir -p $WPSDIR
mkdir -p $PROCSDIR
cp    $WRFTOWPSPATH $WPSDIR/wrf_to_wps.tar
cp    $WRFUTILDIR/vtables/* $WPSDIR/
mkdir -p $WRFDIR
cp -r $WRFUTILDIR/bin $EXPDIR
cp -r $WRFUTILDIR/lib $EXPDIR
mkdir -p $NAMELISTDIR
cp    $WRFUTILDIR/namelists/$MODEL_CONF_SET/* $NAMELISTDIR/
mkdir -p $HISTDIR
mkdir -p $PERTDIR
cp    $PERTMETEMPATH $PERTDIR/pert_met_em.tar
cp    $PERTMETEMPATH $WRFDIR/pert_met_em.tar

#Create tar files containing the WRF, WPS and WRFDA required executables and additional files.
tar -h --dereference -cf $WPSDIR/wps.tar   -C $WPSPATH/ ./
tar -h --dereference -cf $WRFDIR/wrf.tar   -C $WRFPATH/run/ ./

