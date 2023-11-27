#ifndef MULTILAYERABOBATRON_VIEW_TRAININGSCHEDULE_H_
#define MULTILAYERABOBATRON_VIEW_TRAININGSCHEDULE_H_

#include <QWidget>

#include "../Controller/MLPController.h"
#include "../Controller/SettingsController.h"

namespace Ui {
class TrainingSchedule;
}

class TrainingSchedule : public QWidget {
  Q_OBJECT

 public:
  explicit TrainingSchedule(s21::SettingsController& s_controller,
                            s21::MLPController& m_controller,
                            QWidget* parent = nullptr);
  ~TrainingSchedule();

 private:
  size_t GetMLPIndex();
  Ui::TrainingSchedule* ui;
  s21::SettingsController& s_controller_;
  s21::MLPController& m_controller_;
};

#endif  // MULTILAYERABOBATRON_VIEW_TRAININGSCHEDULE_H_
